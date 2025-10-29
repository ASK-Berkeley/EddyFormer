from nsm.typing import *

from nsm.flow import Flow
from nsm.model import Model
from nsm.writer import Writer

from tqdm import tqdm

# ---------------------------------------------------------------------------- #
#                                   PROTOCOL                                   #
# ---------------------------------------------------------------------------- #

class Forward(Protocol):

  def __call__(self, params: PyTree, data: DataType) -> Model.Output:
    """
      Forward pass of the model.
    """

class GradFn(Protocol):

  def __call__(self, params: PyTree, batch_data: DataType) -> Tuple[PyTree, ...]:
    """
      Gradient step of the model.
    """

class TrainFn(Protocol):

  def __call__(self, flow: Flow, model: Model, writer: Writer):
    """
      Training function.
    """

class EvalFn(Protocol):

  def __call__(self, params: PyTree) -> PyTree:
    """
      Evaluation function.
    """

# ---------------------------------------------------------------------------- #
#                                    TRAINER                                   #
# ---------------------------------------------------------------------------- #

@dataclass
class Trainer:

  epoch: int
  iteration: Maybe[int]
  learning_rate: float
  scheduler: optax.Schedule
  optimizer: optax.GradientTransformation

  batch_size: int
  vmap_batch: Maybe[int]
  batch_sharding: bool
  num_cpu_proc: Maybe[int]

  window: Maybe[int]
  gradient_clip: Maybe[float]

  @struct.dataclass
  class State:
    prng: Array
    variable: PyTree
    optim: optax.OptState

  def init(self, flow: Flow, model: Model, prng: Array) -> State:
    """
      Initialize the training state and model parameters.

      Args:
        flow: Flow instance.
        model: Model instance.
        prng: Pseudo random key.
    """
    prng, subkey = jrand.split(prng)
    u, *_ = next(iter(flow.dataset("test")))

    print(model.tabulate(subkey, flow, u.ic, None))
    variable = model.init(subkey, flow, u.ic, None)

    optim = self.optimizer.init(variable["params"])
    return self.State(prng, variable, optim)

  def loss(self, flow: Flow, model: Model, params: PyTree, ic: DataType, ut: Grid):
    """
      Data loss.
    """
    out = model.apply({ "params": params }, flow, ic, out=ut, return_aux=config.debug)
    u = out.u.eval(ut.resolution) if isinstance(out.u, SEM) else out.u

    metric = { "loss": flow.loss_data(u, ut) }
    for key, value in utils.flatten(out.aux).items():
      metric[f"aux/{key}"] = value

    return metric["loss"], (out, metric)

  def step(self, flow: Flow, model: Model, state: State,
           batch_data: Flow.Data) -> Tuple[PyTree, ...]:
    """
      Train the model by minimizing the data loss. The training data is
      a batch of trajectories. Each slice is a tuple of solution at the
      current time, and the corresponding numerical solution.

      Args:
        flow: Flow instance.
        model: Model instance.
        state: Training state.
        batch_data: Training data.
    """
    variable = state.variable
    params = variable["params"]

    prng, subkey = jrand.split(state.prng)
    del subkey # random key is unused

    grad, metric = self.grad(flow, model)(params, batch_data)
    stats = {
      "loss": jnp.mean(metric["loss"]),
      "norm": (norm:=utils.norm(grad)),
    }
    for key, values in metric.items():
      metric[key] = { f"t={t * flow.t}": x
        for t, x in enumerate(values, 1) }

    if self.gradient_clip is not None:
      clip = jnp.array(self.gradient_clip, norm.dtype)
      grad = jax.tree.map(F.partial(lax.cond, norm < clip,
                   lambda x: x, lambda x: x / norm * clip), grad)

    if config.debug:
      aux = dd(lambda: {})

      for name, values in grad.items():
        cat = name.split("_", 1)[0] # id

        for key, value in utils.flatten(values).items():
          aux[cat][f"{name}/{key}"] = jnp.linalg.norm(jnp.ravel(value))

      metric["norm"] = {}
      for cat, norms in aux.items():
        xs = jnp.array([x for x in norms.values()])
        metric["norm"][cat] = jnp.linalg.norm(xs)
        metric[f"norm/{cat}"] = norms

    with jax.numpy_dtype_promotion("standard"): # optax is not strict
      updates, optim = self.optimizer.update(grad, state.optim, params)
      variable["params"] = optax.apply_updates(params, updates)

    return self.State(prng, variable, optim), stats, metric

# --------------------------------- PROTOCOL --------------------------------- #

  def grad(self, flow: Flow, model: Model) -> GradFn:
    """
      Make gradient calculation function.

      Args:
        flow: Flow instance.
        model: Model instance.
    """
    if self.window == 1:

      def batch_grad(params: PyTree, batch_data: Flow.Data):
        """
          One-step training.
        """
        grad_t = jax.grad(F.partial(self.loss, flow, model), has_aux=True)
        grad, (_, metric) = utils.vmap(F.partial(grad_t, params), self.vmap_batch)(batch_data[0].ic, batch_data[0].ut)

        grad, metric = jax.tree.map(F.partial(jnp.mean, axis=0), (grad, metric))
        if self.batch_sharding: grad, metric = lax.pmean((grad, metric), axis)
        return grad, { key: value[None] for key, value in metric.items() }

    else:

      def batch_loss(params: PyTree, batch_ic: Grid, batch_ut: Grid) -> PyTree:
        loss = jax.checkpoint(F.partial(self.loss, flow, model, params))
        return utils.vmap(loss, self.vmap_batch)(batch_ic, batch_ut)

      @F.partial(jax.grad, has_aux=True)
      def batch_grad(params: PyTree, batch_data: List[Flow.Data]):
        """
          N-step training.
        """
        device_batch = self.batch_size // jax.local_device_count()
        logging.info(f"Device batch size: {device_batch}")

        @struct.dataclass
        class ScanState:

          u: DataType
          loss: PyTree

        @jax.checkpoint
        def next(carry: ScanState, batch_data_t: Flow.Data_t) -> Tuple[ScanState, PyTree]:
          """
            Auto-regressively roll out the model and accumulate loss at each step.
          """
          loss, (out, metric) = batch_loss(params, carry.u, batch_data_t.ut)
          return ScanState(out.u, carry.loss + jnp.mean(loss)), jax.tree.map(jnp.mean, metric)

        xs = jax.tree.map(lambda *xs: jnp.stack(xs), *batch_data)
        scan, metric = lax.scan(next, ScanState(batch_data[0].ic, 0.), xs)

        loss = scan.loss / len(batch_data)
        result = loss, metric
        if self.batch_sharding:
          result = lax.pmean(result, axis)
        return result

    if self.batch_sharding:
      mesh = Mesh(jax.devices("gpu"), axis:="batch")

      batch_grad = shard_map(batch_grad, mesh=mesh, in_specs=(P(None), P(axis)), out_specs=P(None))
      logging.info(f"Sharded over {jax.local_device_count()} device(s): {jax.local_devices()}")

    return batch_grad

# ---------------------------------------------------------------------------- #
#                                     TRAIN                                    #
# ---------------------------------------------------------------------------- #

  @classmethod
  def make_train(cls, cfg: ConfigDict) -> TrainFn:
    """
      Make training functions.
    """
    def train(flow: Flow, model: Model, writer: Writer):
      """
        Training loop. First, load or initialize the training state. Then
        compile the training iteration and start the training loop.

        Args:
          cfg: Training configuration.
          flow: Flow instance.
          model: Model instance.
          writer: Writer instance.
      """
      trainer = cls(**cfg.train)
      prng = jrand.PRNGKey(cfg.seed)

      try:
        it, state = writer.load_checkpoint(cfg.log.load_ckpt_it)
      except FileNotFoundError:
        prng, subkey = jrand.split(prng)
        it, state = 0, trainer.init(flow, model, subkey)

      step = F.partial(trainer.step, flow, model)
      forward = F.partial(model.apply, flow=flow)

      if cfg.compile:
        step = utils.jit(step, donate_argnums=0)
        forward = jax.jit(forward, donate_argnames="ic")

      if config.debug:
        from nvtx import annotate
        step = annotate("step", "blue")(step)
        forward = annotate("forward", "red")(forward)

      eval_fn = cls.make_eval(forward, flow)

# ----------------------------------- STEP ----------------------------------- #

      last_save = time.time()
      for epoch in range(trainer.epoch):

        prng, subkey = jrand.split(prng)
        dataset = flow.dataset("train", subkey, trainer.window, 64, trainer.num_cpu_proc)
        dataloader = utils.batched(dataset, trainer.batch_size)

        if config.tqdm: dataloader = tqdm(dataloader, f"Train epoch {epoch}", initial=it)
        for it, batch_data in enumerate(dataloader, it):

          if cfg.log.test_period and it % cfg.log.test_period == 0:

            metric = eval_fn(state.variable)
            writer.write(it, metric:=jax.device_get(metric))
            logging.info(f"Iter {it}: {metric}")

          now = time.time()
          if now - last_save >= cfg.log.ckpt_period:
            writer.save_checkpoint(it, state)
            last_save = now

          state, stats, metric = step(state, batch_data)
          stats, metric = jax.device_get((stats, metric))
          writer.write(it, dict(stats=stats) | metric)

          if config.tqdm: dataloader.set_postfix(stats)
          else: logging.info(f"Epoch {epoch}: {it}it - {stats}")

          if trainer.iteration and it >= trainer.iteration: break

        else:
          it += 1
          continue
        break

      writer.save_checkpoint(it, state)
      writer.finish()

    return train

# ---------------------------------------------------------------------------- #
#                                     EVAL                                     #
# ---------------------------------------------------------------------------- #

  @classmethod
  def make_eval(cls, f: Forward, flow: Flow) -> EvalFn:
    """
      Make evaluation steps.

      Args:
        f: Forward model.
        flow: Flow instance.
    """
    def roll(variable: PyTree, data: Flow.Data) -> PyTree:
      """
        Rollout the model on the test set. Returns a tree of lists,
        where each index corresponds to the rollout time step.

        Args:
          variable: Model parameters.
          solution: Reference trajectory.
      """
      u = data[0].ic

      metric = dd(lambda: [])
      for i in range(len(data)):

        if u.value.shape[-1] == data[i].ic.value.shape[-1]:
          out = f(variable, ic=u)
        else:
          out = f(variable, ic=data[i].ic)

        u = out.u if isinstance(out.u, Grid) else out.u.eval(data[i].ut.resolution)
        metric_t = flow.metric(u, data[i].ut)

        for key, value in metric_t.items():
          metric[key].append(jax.tree.map(float, value))

      return dict(metric)

    def eval(variable: PyTree) -> Tuple[PyTree, PyTree]:
      """
        Evaluate the model on the test set.

        Args:
          variable: Model parameters.
      """
      dataset = flow.dataset("test")
      if config.tqdm: dataset = tqdm(dataset, "Test")

      metrics = dd(lambda: [])
      for data in dataset:

        for key, values in roll(variable, data).items():
          for t, value in enumerate(values):
            metrics[f"{key}_avg"].append(value)
            metrics[f"{key}/{t}"].append(value)

      mean = lambda *x: np.mean(x).item()
      metrics = { key: jax.tree.map(mean, *value)
              for key, value in metrics.items() }

      return metrics

    return eval
