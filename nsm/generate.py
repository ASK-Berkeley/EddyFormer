from nsm import *
from nsm.typing import *

from nsm.flow import Flow

from tqdm import tqdm
from random import Random

def get_config() -> ConfigDict:

  return ConfigDict({
    # dataset split
    "label": "test",
    # random seed
    "seed": 19260817,
    # solver batch
    "batch_size": 1,
    # total simulation time
    "t": config_dict.placeholder(float),
    # recording time interval
    "dt": config_dict.placeholder(float),
  })

def main(cfg: ConfigDict, flow: Flow):
  cfg = cfg.copy_and_resolve_references()
  os.makedirs(flow.base_dir, exist_ok=True)

  def init(i: int) -> Tuple[Grid, str]:
    rng = Random(f"{cfg.label}-{cfg.seed}-{i}")
    key = jrand.PRNGKey(seed:=rng.getrandbits(32))
    fname = f"{flow.base_dir}/{cfg.label}.{seed:08x}.npy"
    ic, _ = flow.ic.init_with_output(key, 1)
    assert not os.path.exists(fname), f"file {fname} already exists"
    logging.info(f"Saving initial condition to {fname}...")
    np.save(fname, { "dt": cfg.dt, "u": ic.value })
    return ic, fname

  @jax.pmap
  def step(ic: Grid) -> Tuple[Grid, Array]:
    logging.info(f"Solver step shape: {ic.value.shape}")
    u = flow.solve(ic, jnp.array(cfg.dt or flow.T))
    k = flow.resolution[0] // flow.ic.resolution[0]
    return u, u.value[(slice(None, None, k), ) * u.ndim]

  logging.info(f"Sampling {cfg.batch_size} parameters ...")
  ics, fnames = zip(*map(init, range(cfg.batch_size)))
  batch_u = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *ics)

  def save(us: np.ndarray, t: int):
    for u, fname in zip(us, fnames):

      if not np.all(np.isfinite(u)):
        logging.error(f"NaNs detected!")

      else:
        logging.info(f"Saving step {t} to {fname} ...")
        data = np.load(fname, allow_pickle=True).item()
        data["u"] = np.append(data["u"], u[None], axis=0)
        np.save(fname, data)

  logging.info(f"Running solver on {flow.resolution} ...")
  num_steps = range(int(cfg.t / cfg.dt)) if cfg.t else I.count()
  if config.tqdm: num_steps = tqdm(num_steps, desc="Time steps")

  for t in num_steps:

    batch_u, us = step(batch_u)
    save(us, t)
