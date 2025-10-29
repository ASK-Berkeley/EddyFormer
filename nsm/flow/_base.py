from nsm.typing import *

class Flow(Protocol):

  ndim: int     # domain dimension
  odim: int     # output dimension

  t: float      # time step
  base_dir: str # dataset directory

  def process(self, ic: Grid, out: DataType, aux_data: PyTree) -> Tuple[DataType, Maybe[Grid]]:
    """
      Pre-processes the initial condition.

      Args:
        ic: Initial condition.
        out: Target datatype.
        aux_data: Mutable auxiliary data.
    """

  def project(self, u: DataType, u0: Maybe[Grid], out: DataType) -> DataType:
    """
      Transformation that post-process predictions.

      Args:
        u: Predicted solution.
        u0: Coarse solver guess.
        out: Target datatype.
    """

  def metric(self, u: Grid, ut: Grid) -> PyTree:
    """
      Metrics for the predicted solution.

      Args:
        u: Predicted solution.
        ut: Reference solution.
    """

  def loss_data(self, u: Grid, ut: Grid) -> Array:
    """
      Data loss for the flow.

      Args:
        u: Predicted solution.
        ut: Reference solution.
    """

# ---------------------------------- DATASET --------------------------------- #

  @struct.dataclass
  class Data_t:

    t: Array
    ic: Grid
    ut: Grid

  Data = List[Data_t]

  def dataset_files(self, split: str) -> Sequence[Any]:
    """
      Files of a given dataset split.

      Args:
        split: Name of the split.
    """

  def dataset_transform(self, fname, window: Maybe[int], prng: Maybe[Array]) -> Data:
    """
      Load and transform each data sample.

      Args:
        split: Name of the split.
        i: Index of the data sample.
    """

  def dataset(self, split: str,
              prng: Maybe[Array] = None,
              window: Maybe[int] = None,
              max_size: int = 1,
              num_proc: int = 1) -> Iterator[Data]:
    """
      Load solution dataset from disk. Returns an iterator that yields
      a batch of trajectories, each of which consists of the numerical
      solution at each time steps.

      Args:
        split: Data split (e.g., train, test, ...).
        prng: Shuffle the dataset with PRNG, if provided.
        window: Sampled window size (`prng` required).
        num_proc: Number of CPU workers to load the dataset.
    """
    idx = jnp.arange(len(files:=self.dataset_files(split)))
    if prng is not None: idx = jrand.permutation(prng, idx)
    f = lambda i: self.dataset_transform(files[i], window, jrand.fold_in(prng, i) if window else None)
    return utils.imap(f, idx, max_size=max_size, num_proc=num_proc, block=False) # dataloader shouldn't block main thread
