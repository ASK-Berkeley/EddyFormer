from nsm._typing import *
from nsm import _utils as utils

from flax.struct import dataclass
from functools import cached_property

from nsm._basis import Basis, Chebyshev, Legendre
from nsm._basis._element import make_element

DataType = Union["Grid", "SEM"]

# ---------------------------------------------------------------------------- #
#                                   GRID DATA                                  #
# ---------------------------------------------------------------------------- #

@dataclass
class Grid:

  """
    An uniform grid.
  """

  value: Array
  size: Array

  @property
  def ndim(self) -> int:
    return len(self.size)

  @property
  def resolution(self) -> Shape:
    return self.value.shape[:self.ndim]

  @property
  def coords(self) -> Array:
    return utils.grid(*self.resolution) * self.size

  def interp(self, out: DataType, method: Union[str, List[str]]) -> DataType:
    """
      Interpolate grid values to a target datatype.

    Args:
      out: Target datatype.
      method: Interpolation method along each axis.
    """
    if isinstance(method, str):
      method = [method] * self.ndim

    def interp1d(value: Array, xs: Union[int, Array], method: str) -> Array:
      """
      Interpolate from 1D grid value to a target points.

      Args:
        value: Values on a uniform grid along the first axis.
        xs: Resolution or an array normalized by the domain size.
        method: Interpolation method. One of ["pbc", "linear"].
      """
      if method == "pbc":
        coef = jnp.fft.rfft(value[:-1], axis=0, norm="forward")

        if isinstance(xs, int): # uniform grid; use iFFT
          value = jnp.fft.irfft(coef, xs - 1, axis=0, norm="forward")
          pad = F.partial(jnp.pad, pad_width=(0, 1), mode="wrap")
          return jnp.apply_along_axis(pad, axis=0, arr=value)

        if isinstance(xs, Array): # arbitrary; use matmul
          k = 2 * jnp.pi * jnp.arange(len(coef), dtype=float) * xs
          f = jnp.exp(1j * k.astype(complex)).at[..., 1:-1].mul(2)
          return jnp.tensordot(f.real, coef.real, axes=1) \
               - jnp.tensordot(f.imag, coef.imag, axes=1)

      if method == "linear":
        from jax.scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator([jnp.linspace(0, 1, len(value))], value)

        if isinstance(xs, int):
          xs = utils.grid(xs)

        xs = xs.squeeze(-1)
        return utils.reshape(f(jnp.ravel(xs)), xs.shape, end=1)

      raise ValueError(f"invalid interpolation {method=}")

    if isinstance(out, Grid):

      value = self.value
      for n in range(self.ndim):

        # iteratively interpolate along each dimension `n`
        value = interp1d(value, out.resolution[n], method[n])
        value = jnp.moveaxis(value, 0, self.ndim - 1) # roll

    # assert self.size == out.size
      return Grid(value, self.size)

    if isinstance(out, SEM):
      xs = utils.reshape(out.coords / self.size, out.mesh, self.ndim, -1)

      value = self.value
      for n in range(self.ndim):

        # interpolate at each collocation points. `idx` is the
        # index of the elements along the `n`'th dimension.
        idx = [slice(None) if i == n else 0 for i in range(self.ndim)]
        value = interp1d(value, xs[tuple(idx * 2 + [n])][..., None], method[n])

        # roll the output. The interpolated values have shape `(mode, mesh)`,
        # which are moved to the middle (`ndim - 1`) and the end (`ndim + n`) of
        # the dimensions. After `ndim` iterations, all axes are ordered correctly.
        value = jnp.moveaxis(value, (0, 1), (self.ndim - 1, self.ndim + n))

      return out.new(utils.reshape(value, -1, out.ndim, out.ndim * 2))

    raise ValueError(f"invalid output type {out=}")

# ---------------------------------------------------------------------------- #
#                               SPECTRAL ELEMENT                               #
# ---------------------------------------------------------------------------- #

@dataclass
class SEM:

  """
    Spectral element expansion. The sub-domain partition is
    given by the `mesh` attribute. The spectral coefficients
    of each element is stored in the first channel dimension, 
    whose size must equal to the number of elements.
  """

  T_: str = struct.field(False)

  # Mesh

  size: Array
  mesh: Shape = struct.field(False)

  # Data

  mode_: Maybe[Shape] = None
  nodal: Maybe[Array] = None

  @property
  def ndim(self) -> int:
    return len(self.mesh)

  @property
  def mode(self) -> Shape:
    if self.mode_: return self.mode_
    return self.nodal.shape[:self.ndim]

  @property
  def use_elem(self) -> bool:
    return self.T_.endswith("elem")

  @cached_property
  def T(self) -> Type[Basis]:
    """
      Basis type on each dimension.
    """
    if self.T_.startswith("cheb"): T = Chebyshev
    if self.T_.startswith("leg"): T = Legendre

    try:
      return make_element(T) if self.use_elem else T
    except NameError:
      raise ValueError(f"unknown basis type {self.T_}")

  def to(self, mode: Shape) -> "SEM":
    """
      Resample to another mode.

      Args:
        mode: Number of modes.
    """
    value = self.nodal

    for n in range(self.ndim):
      coef = jnp.apply_along_axis(self.T.modal, n, value, min(mode[n], self.mode[n]))
      value = jnp.apply_along_axis(self.T.nodal, n, coef, mode[n])

    return self.new(value)

  def at(self, *xs: Array) -> Array:
    """
      Evaluate on rectilinear grids.

      Args:
        xs: Coordinate of each dimension.
    """
    value = utils.reshape(self.nodal, self.mesh, self.ndim, self.ndim + 1)

    for n in range(self.ndim):
      x = xs[n] / self.size[n]

      # indices of each global coordinate `x`
      idx = jnp.floor(x * self.mesh[n]).astype(int)
      idx = jnp.minimum(idx, self.mesh[n] - 1) # endpoint

      # coefficients where each `x` belongs
      coef = self.T.modal(value)
      coef = jnp.take(coef, idx, axis=self.ndim)

      # global coordinate to local coordinate
      ys = (x[..., None] - utils.grid(self.mesh[n], mode="left")[idx]) * self.mesh[n]

      # evaluate at each coordonate and move the output axis to the last dimension
      # after `ndim` iterations, the axes are automatically rolled to the correct order
      value = jax.vmap(self.T.at, (self.ndim, 0), self.ndim - 1)(coef, ys)

    return value

# ---------------------------------- COORDS ---------------------------------- #

  def __len__(self) -> int:
    return math.prod(self.mesh)

  @cached_property
  def grid(self) -> Array:
    axes = [self.T.grid(self.mode[i]).squeeze(1) for i in range(self.ndim)]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

  @cached_property
  def coords(self) -> Array:
    move = lambda xs: xs + self.grid * self.lengths
    return jax.vmap(move, out_axes=self.ndim)(self.origins)

  @cached_property
  def origins(self) -> Array:
    return utils.grid(*self.mesh, mode="left", flatten=True) * self.size

  @cached_property
  def lengths(self) -> Array:
    return self.size / jnp.array(self.mesh, dtype=float)

# --------------------------------- DATA TYPE -------------------------------- #

  def new(self, nodal: Array) -> "SEM":
    return SEM(self.T_, self.size, self.mesh, nodal=nodal)

  def eval(self, resolution: Shape) -> Grid:
    xs = [utils.grid(n).squeeze(-1) * s for n, s in zip(resolution, self.size)]
    return Grid(self.at(*xs), self.size)
