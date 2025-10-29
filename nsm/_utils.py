from nsm._typing import *

def norm(xs: PyTree) -> Array:
  """
    Norm of the PyTree.

    Args:
      xs: PyTree of vectors.
  """
  return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(xs)))

def flatten(xs: PyTree) -> Dict[str, Array]:
  """
    Flatten the PyTree into a dictionary, whose
    keys are the directory of the give tree.

    Args:
      xs: PyTree to be flattened.
  """
  if isinstance(xs, Sequence):
    return { f"{idx}/{key}": value 
            for idx, x in enumerate(xs)
            for key, value in flatten(x).items() }
  if isinstance(xs, Dict):
    return { f"{key1}/{key2}": value
            for key1, x in xs.items()
            for key2, value in flatten(x).items() }

  msg = f"invalid type: {type(xs)}"
  assert isinstance(xs, Array), msg

  return { "": xs }

def reshape(x: Array, shape: Union[int, Shape] = -1,
            start: int = 0, end: Maybe[int] = None) -> Array:
  """
    Reshape certain axis.

    Args:
      x: Array to be reshaped.
      shape: Shape of axes specified by `start` and `end`.
      start: First axis to be reshaped. Default to the first.
      end: Last axis to be reshaped. Default to the last axis.
  """
  end = () if end is None else x.shape[end:]

  if not isinstance(shape, Sequence): shape = shape,
  return x.reshape(x.shape[:start] + tuple(shape) + end)

def batched(iterable: Iterable, batch_size: Maybe[int]) -> Iterable:
  """
    Batched iterator.

    Args:
      dataset: Dataset iterable.
      batch_size: Collate batch size.
  """
  if batch_size is None: return iterable

  def collocate(batch: List[PyTree]) -> PyTree:
    with jax.default_device(jax.devices("cpu")[0]):
      return jax.tree.map(lambda *xs: jnp.stack(xs), *batch)
  return map(collocate, zip(*[iter(iterable)] * batch_size))

def grid(*s: int, mode: str = None, flatten: bool = False) -> Array:
  """
    Uniform grids on [0, 1)^n. If not `flatten`, shape=(*s, len(s));
    else the grid is flattened, i.e., shape=(math.prod(s), len(s)).

    Args:
      s: Size of the grid in each dimension.
      mode: Type of the grid points.
        - `None`: uniformly spaced;
        - "left": exclude endpoint;
        - "cell": centers of rects.
  """
  axes = F.partial(jnp.linspace, 0, 1, endpoint=mode is None)
  grid = jnp.stack(jnp.meshgrid(*map(axes, s), indexing="ij"), -1)

  if mode == "cell": grid += .5 / jnp.array(s, float)
  if flatten: return grid.reshape(-1, len(s))

  return grid

# ---------------------------------- METRICS --------------------------------- #

def corr(x: Array, y: Array) -> Array:
  """
    Pearson correlation.
  """
  return jnp.sum(x * y) / jnp.linalg.norm(jnp.ravel(x)) \
                        / jnp.linalg.norm(jnp.ravel(y))

def err(x: Array, y: Array) -> Array:
  """
    Relative l2 error.
  """
  return jnp.linalg.norm(jnp.ravel(x - y)) \
       / jnp.linalg.norm(jnp.ravel(y))

def mae(x: Array, y: Array) -> Array:
  """
    Mean absolute error.
  """
  return jnp.mean(jnp.abs(x - y))

# ---------------------------------------------------------------------------- #
#                                      JAX                                     #
# ---------------------------------------------------------------------------- #

def jit(f, **options):
  """
    JIT function with cost analysis on the first run. Keep in mind that loops
    are not taken in to account correctly (which means with `vmap_batch` set,
    the flop and memory estimation results are not reliable).

    Args:
      f: Function to be jitted.
      options: Arguments for `jax.jit`.
  """
  f = jax.jit(f, **options)
  f_compiled = {}

  @F.wraps(f)
  def call(*args, **kwargs):
    key = hash(jax.tree.structure((args, kwargs)))

    if key in f_compiled:
      g = f_compiled[key]
    
    else:
      print("=" * 80)

      print(f"compling function {f} ......")
      g = f.lower(*args, **kwargs).compile()
      f_compiled[key] = g

      print(g.cost_analysis())
      print("=" * 80)

    return g(*args, **kwargs)

  return call

def timeit(f, label: Maybe[str] = None, result: Maybe[List] = None):
  """
    Time the execution of a function. The input function is compiled (jitted) in
    advacnce. Returns the mean and standard deviation of its execution time.

    Args:
      f: Function to be timed.
      label: Custom tag for f (`f.__str__` by default).
      result: List to store the execution time. This is
              only used when the function is not jitted.
  """
  f_jit = jit(f)
  label = label or str(f)

  t0, count = None, 0
  def record():
    nonlocal t0, count
    t0 = time.time(); count += 1

  def log(xs: PyTree, ts = result if result is not None else []):
    jax.tree.map(jax.block_until_ready, xs) # synchronize results

    nonlocal t0, count
    dt = time.time() - t0

    if count <= 3:
      print(f"%time {label}: {dt:.3f} sec ({count}th run)")
    else:
      ts.append(dt)
      print(f"%time {label}: {np.mean(ts):.3f} ± {np.std(ts):.3f} sec ({count} runs)")

  def call(*args, **kwargs):

    jax.debug.callback(record)
    xs = f_jit(*args, **kwargs)
    jax.debug.callback(log, xs)

    return xs

  return call

# ------------------------------------ CPU ----------------------------------- #

from queue import Queue, Empty, Full
from threading import Thread, Lock, Barrier

def _imap_worker(f: Callable, args: Iterator, keys: Sequence[str],
                 device: Any, init: Barrier, mutex: Lock, q: Queue):
  """
    Worker thread. The worker keeps processing `xs` on its
    own device, unless blocked by the queue size. The `init`
    barrier is released after queue `ys` is full for the first
    time. After the worker finishes its jobs, it signals the
    main thread by putting its device into the queue.
  """
  buf = Queue(-1)
  block = False
  while True:

    with mutex:
      try:
        arg, kwarg = next(args), {}
        if (n:=len(keys)):
          arg, value = arg[:-n], arg[-n:]
          kwarg = dict(zip(keys, value))
      except StopIteration: break

      try: q.put(buf, block)
      except Full: block = True

    try:
      with jax.default_device(device):
        buf.put(f(*arg, **kwarg))
    except Exception as e:
      if init: init.reset()
      raise e

    if block and init:
      init.wait()
      init = None
      q.put(buf, True)

  if init: init.wait()

  q.put(device)

def imap(f: Callable,
         *args: Iterable,
         num_proc: int = -1,
         max_size: int = 64,
         backend: str = "cpu",
         block: bool = True,
         **kwargs: Iterable,
         ) -> Iterable:
  """
    CPU-based parallel map. The workers lazily map a list of
    arguments, similar to `Pool.imap`, but the workers are
    thread-based and therefore are compatible with JAX.

    Args:
      f: Function to be mapped.
      num_proc: Number of workers.
      max_size: Maximum buffer size.
      backend: JAX supported backend.
      block: Should I expect it to block?
      args: Iterable of position args.
      kwargs: Iterable of keyword args.
  """
  devices = jax.devices(backend)
  if num_proc == -1:
    num_proc = len(devices)
  else:
    assert num_proc <= len(devices), \
      f"insufficient {backend} devices: len({devices}) = {len(devices)} < {num_proc}." \
       "try setting `xla_force_host_platform_device_count` in XLA_FLAGS to a proper value."
    devices = devices[:num_proc]

  args = iter(zip(*args, *kwargs.values()))
  init, mutex = Barrier(num_proc + 1), Lock()

  q = Queue(max_size)
  workers = {
    device: Thread(
      target=_imap_worker,
      daemon=True,
      args=(f, args, kwargs.keys(), device, init, mutex, q),
    ) for device in devices
  }
  for worker in workers.values():
    worker.start()

  init.wait()
  while workers:
    try:
      x = q.get(block)
      if isinstance(x, jax.Device):
        workers.pop(x).join()
      else:
        assert isinstance(x, Queue)
        yield x.get(block)
    except Empty:
      block = True
      logging.info(f"Queue for {f} blocked!")

# ------------------------------------ GPU ----------------------------------- #

def vmap(f, batch_size: Maybe[int] = None, **options):
  """
    Vectorizing map. Extending `jax.vmap` by allowing loop-based
    map over a batch of data chunks.

    Args:
      f: Function to be vmapped.
      batch: Size of the chunk. No batching if `None`.
      options: Additional arguments for `jax.vmap`.
  """
  f = jax.vmap(f, **options)
  if not batch_size: return f

  def reshape(xs: Array) -> Array:
    return xs.reshape(-1, batch_size, *xs.shape[1:])

  @F.wraps(f)
  def call(*args, **kwargs):
    args = jax.tree.map(reshape, args)
    kwargs = jax.tree.map(reshape, kwargs)

    f_wrap = lambda arg: f(*arg[0], **arg[1])
    result = lax.map(f_wrap, (args, kwargs))
    return jax.tree.map(jnp.concatenate, result)

  return call
