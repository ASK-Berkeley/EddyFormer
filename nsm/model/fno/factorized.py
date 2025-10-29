from nsm.typing import *
from .._base import Model

from .zongyi import kernel_init

def ffn(odim: Union[int, Shape], activation: str, x: Array, hdim: Maybe[int] = None) -> Array:
  return nn.DenseGeneral(odim)(getattr(nn, activation)(nn.Dense(hdim or x.shape[-1] * 4)(x)))

class SpectralConv(nn.Module):

  """
    Factorized spectral convolution generalized
    to support arbitrary number of dimensions.
  """

  ndim: int
  odim: int

  mode: Maybe[Shape]
  kernel_init: str = "ffno"
  use_complex: bool = True

  @nn.compact
  def factor(self, coef: Array, n: int) -> Array:
    """
      Factorized spectral convolution along axis `n`.

      Args:
        coef: Fourier coefficients.
        n: Axis to convolve along.
    """
    ns = "".join(map(str, range(self.ndim)))
    einsum = f"{ns}...i, {n}io -> {ns}...o"

    init = F.partial(kernel_init, self.kernel_init)
    mode = self.mode or coef.shape[:self.ndim]

    Wn = self.param(f"W_{n}", init, shape:=(mode[n], coef.shape[-1], self.odim))
    if self.use_complex: Wn = lax.complex(Wn, self.param(f"W_{n}i", init, shape))

    if coef.shape[n] < mode[n]: Wn = Wn[:coef.shape[n]]
    else: coef = jnp.take(coef, jnp.r_[:mode[n]], axis=n)

    if config.debug:
      logging.info(f"Axis {n}: {einsum} with {coef.shape}, {Wn.shape}")

    return jnp.einsum(einsum, coef, Wn)

  def __call__(self, x: Array, n: Maybe[int] = None) -> Array:
    """
      Factorized spectral convolution.

      Args:
        x: Input on uniform grids.
        n: Axis to convolve along.
    """
    if n is not None:
      factor = self.factor(jnp.fft.rfft(x, axis=n), n)
      return jnp.fft.irfft(factor, x.shape[n], axis=n)

    return sum(self(x, n) for n in range(self.ndim))

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

class FFNO(nn.Module, Model):

  hdim: int
  odim: int
  depth: int

  mode: Shape
  activation: str

  # @nn.checkpoint
  def layer(self, x: Array) -> Array:
    conv = SpectralConv(len(self.mode), self.hdim, self.mode)
    return x + ffn(self.hdim, self.activation, conv(x))

  @nn.compact
  def forward(self, ϕ: Grid, aux_data: Maybe[PyTree]) -> Grid:
    """
      Forward pass of Factorized FNO.
    """
    del aux_data

    x = jnp.concatenate([ϕ.coords, ϕ.value], axis=-1)
    x = ffn(self.hdim, self.activation, x)

    for _ in range(self.depth):
      x = self.layer(x)

    x = ffn(self.hdim, self.activation, x)
    return Grid(nn.Dense(self.odim)(x), ϕ.size)
