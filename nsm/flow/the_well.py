from nsm import *
from nsm.typing import *

from nsm.flow import Flow
from yaml import safe_load
from h5py import Dataset, File

DATASET_DIM = {
  "MHD_64": 3,
  "shear_flow": 2,
  "rayleigh_benard": 2,
  "rayleigh_taylor_instability": 3,
}

def _canonize(field: Dataset, n: int, it: slice, stats: Maybe[Dict]) -> Array:
  """
    Reconstruct the full field and normalize it.

    Args:
      field: A dataset in The Well.
      n: Index of the trajectory sample.
      it: Index of the time step.
      stats: Normalization statistics.
  """
  ix = ()

  if field.attrs["sample_varying"]: ix += n,
  if field.attrs["time_varying"]: ix += it,

  resize = (slice(None),) if field.attrs["time_varying"] else (None,)
  for v in field.attrs["dim_varying"]: resize += (slice(None),) if v else (None,)

  data = jnp.array(field[ix])[resize]
  if stats is not None:
    data -= jnp.array(stats["mean"])
    data /= jnp.array(stats["std"])
  return data

UNKNOWN = object()

class TheWell(Flow):

  name: str
  field_names: List = UNKNOWN

  eps: float = 1e-7
  n_steps_input: int = 4

  @property
  def ndim(self) -> int:
    return DATASET_DIM[self.name]

  @F.cached_property
  def stat(self) -> Dict[str, Union[float, List[float]]]:
    stat = safe_load(open(f"{self.base_dir}/stats.yaml"))
    return { key: {
      "mean": jnp.array(stat["mean"][key]),
      "std": jnp.array(stat["std"][key]),
    } for key in stat["mean"] }

  def __init__(self, base_dir: Maybe[str], name: str):
    self.base_dir, self.name = base_dir or f"data/the_well/datasets/{name}", name
    self.t = 1 # `t` here only means time step; we don't condition on the time

# ----------------------------------- FLOW ----------------------------------- #

  def process(self, ic: Grid, out: DataType, aux_data: PyTree) -> Tuple[DataType, None]:
    """
      Interpolate the input fields.
    """
    del aux_data
    bcs = "pbc"

    if self.name == "rayleigh_benard": bcs = ["pbc", "linear"]
    if self.name == "rayleigh_taylor_instability": bcs = ["pbc", "pbc", "linear"]

    return ic.interp(out, bcs), None

  def project(self, u: DataType, u0: None, out: Maybe[DataType] = None) -> DataType:
    """
      Evaluate on target grid.
    """
    return Grid(u.eval(out.resolution).value if isinstance(u, SEM) else u.value, u.size)

  def metric(self, u: Grid, ut: Grid) -> PyTree:
    """
      VRMSE of each field.
    """
    metric = { "mse": {}, "nrmse": {}, "vrmse": {} }
    for n, field_name in enumerate(self.field_names):
      if "@" in field_name:
        name, i = field_name.split("@"); i = tuple(map(int, i))
        mean, std = self.stat[name]["mean"][i], self.stat[name]["std"][i]
      else:
        mean, std = self.stat[field_name]["mean"], self.stat[field_name]["std"]
      y_pred, y = u.value[..., n] * std + mean, ut.value[..., n] * std + mean
      metric["mse"][field_name] = (mse:=jnp.mean((y_pred - y) ** 2))
      metric["nrmse"][field_name] = jnp.sqrt(mse / (jnp.mean(y ** 2) + self.eps))
      metric["vrmse"][field_name] = jnp.sqrt(mse / (jnp.mean((y - y.mean()) ** 2) + self.eps))

    return metric

  def loss_data(self, u: Grid, ut: Grid) -> Array:
    return jnp.mean((u.value - ut.value) ** 2) # mse

# ---------------------------------- DATASET --------------------------------- #

  @F.lru_cache
  def dataset_files(self, split: str) -> List[str]:
    """
      List all trajectories in every `.hfd5` file.
    """
    def unbatch(fname: str) -> Sequence[str]:
      if not fname.endswith(".hdf5"): return []
      with File(dir:=f"{self.base_dir}/data/{split}/{fname}", "r") as file:
        return [f"{dir}:{n}" for n in range(file.attrs["n_trajectories"])]

    logging.info(f"Loading {split} dataset under {self.base_dir}/data ...")
    return sum(map(unbatch, sorted(os.listdir(f"{self.base_dir}/data/{split}"))), [])

  def dataset_transform(self, fname: str, window: Maybe[int], prng: Maybe[Array]) -> Flow.Data:
    """
      Load the dataset and group the trajectory by [:window] -> [window].
    """
    if config.debug:
      logging.info(f"Loading {fname}...")
    fname, n = fname.split(":")

    with File(fname, "r") as file:
      ts = jnp.array(file["dimensions"]["time"])

      it = slice(None)
      if window is not None:
        s = jrand.choice(prng, len(ts) - window - self.n_steps_input + 1).item()
        ts = ts[(it:=slice(s, s + window + self.n_steps_input))]

      inputs, output, field_names = [], [], []
      for order, key in enumerate(["t0_fields", "t1_fields", "t2_fields"]):
        for name, field in file[key].items():
          data = _canonize(field, int(n), it, self.stat[name])
          input = utils.reshape(data, -1, self.ndim + 1)

          if field.attrs["time_varying"]:
            if order == 0:
              field_names.append(name)
              output.append(input[self.n_steps_input:])
            else:
              for i in jnp.argwhere(jnp.ones(data.shape[-order:])):
                field_names.append(f"{name}@{''.join(map(str, i))}")
                output.append(input[self.n_steps_input:, ..., i])
          else:
            input = jnp.repeat(input, len(ts), axis=0)

          inputs.append(input)

      if self.field_names == UNKNOWN:
        self.field_names = field_names

    if self.name == "shear_flow":
      inputs = [jnp.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="wrap") for x in inputs]
      output = [jnp.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="wrap") for x in output]
    if self.name == "MHD_64":
      inputs = [jnp.pad(x, [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0]], mode="wrap") for x in inputs]
      output = [jnp.pad(x, [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0]], mode="wrap") for x in output]
    if self.name == "rayleigh_benard":
      inputs = [jnp.pad(x, [[0, 0], [0, 1], [0, 0], [0, 0]], mode="wrap") for x in inputs]
      output = [jnp.pad(x, [[0, 0], [0, 1], [0, 0], [0, 0]], mode="wrap") for x in output]
    if self.name == "rayleigh_taylor_instability":
      inputs = [jnp.pad(x, [[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]], mode="wrap") for x in inputs]
      output = [jnp.pad(x, [[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]], mode="wrap") for x in output]

    inputs = jnp.concatenate(inputs, axis=-1)
    output = jnp.concatenate(output, axis=-1)

    size = jnp.ones(self.ndim) # TODO: domain
    return [Flow.Data_t(t, Grid(jnp.concatenate(ics, axis=-1), size), Grid(ut, size)) \
       for ics, ut, t in zip(zip(*(inputs[i:] for i in range(self.n_steps_input))), output, ts[self.n_steps_input:])]
