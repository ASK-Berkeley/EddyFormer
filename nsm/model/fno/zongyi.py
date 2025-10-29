from nsm.typing import *
from .._base import Model

def kernel_init_std(method: str, idim: int, odim: int) -> float:
  """
    FNO kernel initialization standard deviation.

    Args:
      method: Init method. One of:
        - "fno": FNO's original approach,
        - "ffno": Factorized-FNO's approach,
        - "epsilon": small constant (1e-7).
      idim: Number of input channels.
      odim: Number of output channels.
  """
  if method == "epslion": return 1e-7

  fan_avg = math.sqrt(2 / (idim + odim))
  if method == "fno": return fan_avg
  if method == "ffno": return fan_avg * 0.1

  raise ValueError(f"invalid init {method = }")

def kernel_init(method: str, prng: Array, shape: Shape) -> Array:
  """
    Fourier kernel initializers. One of the methods in
    `nn.initializers`, or see `kernel_init_std` above.
  """
  idim, odim = shape[-2:]

  try:
    init = getattr(nn.initializers, method)
  except AttributeError:
    std = kernel_init_std(method, idim, odim)
    def init(prng: Array, shape: Shape) -> Array:
      return std * jrand.normal(prng, shape)
  return init(prng, shape)

class SpectralConv(nn.Module):

  """
    Fourier spectral convolution.
  """

  ndim: int
  odim: int

  mode: Maybe[Shape]
  kernel_init: str = "fno"
  use_complex: bool = True

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """
      Full-rank spectral convolution.

      Args:
        x: Input on uniform grids.
    """
    ns = "".join(map(str, range(self.ndim)))
    einsum = f"{ns}...i, {ns}io -> {ns}...o"

    init = F.partial(kernel_init, self.kernel_init)
    coef = jnp.fft.rfftn(x, None, axes:=range(self.ndim))

    if self.mode is not None:
      for n, mode in enumerate(self.mode[:-1]):
        coef = jnp.take(coef, jnp.r_[-mode//2:mode//2], n)
      coef = jnp.take(coef, jnp.r_[:self.mode[-1]//2], -2)

    W = self.param("W", init, size:=coef.shape[:self.ndim] + (x.shape[-1], self.odim))
    if self.use_complex: W += 1j * self.param("Wi", init, size)

    if config.debug:
      logging.info(f"Conv: {einsum} with {coef.shape}, {W.shape}")

    return jnp.fft.irfftn(jnp.einsum(einsum, coef, W), x.shape[:self.ndim], axes)

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

class FNO(nn.Module, Model):

  hdim: int
  odim: int
  depth: int

  mode: Shape
  activation: str

  # @nn.checkpoint
  def layer(self, x: Array) -> Array:
    conv = SpectralConv(len(self.mode), self.hdim, self.mode)
    return getattr(nn, self.activation)(conv(x) + nn.Dense(self.hdim)(x))

  @nn.compact
  def forward(self, ϕ: Grid) -> Grid:
    """
      Forward pass of Fourier Neural Operator.
    """
    act = getattr(nn, self.activation)
    x = jnp.concatenate([ϕ.coords, ϕ.value], axis=-1)

    x = act(nn.Dense(self.hdim * 4)(x))
    x = act(nn.Dense(self.hdim)(x))

    for _ in range(self.depth):
      x = self.layer(x)

    x = act(nn.Dense(self.hdim)(x))
    x = act(nn.Dense(self.hdim * 4)(x))

    x = nn.Dense(self.odim)(x)
    return Grid(x, ϕ.ndim)
