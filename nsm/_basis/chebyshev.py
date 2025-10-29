from nsm._typing import *
from nsm import _utils as utils

from ._base import Basis
class Chebyshev(Basis):

  """
    Shifted Chebyshev polynomials:
      - `Tn(x) = cos(n cos^-1(x))`
      - `Tn^~(x) = Tn(2 x - 1)`
  """

  @classmethod
  def grid(cls, m: int, endpoint: bool = False) -> Array:
    """
      Chebyshev roots and extrema & boundary points.
    """
    return (1 + jnp.cos(theta(m, endpoint))[::-1]) / 2

  @classmethod
  @F.lru_cache
  def fn(cls, n: int) -> Callable:
    """
      Cosine series on a circle.
    """
    @jax.jit
    def call(xs: Array) -> Array:
      m = jnp.arange(n, dtype=float)
      return jnp.cos(m * jnp.arccos(xs * 2 - 1))
  
    return call

# --------------------------------- TRANSFORM -------------------------------- #

  @classmethod
  def modal(cls, vals: Array, m: Maybe[int] = None, endpoint: bool = False) -> Array:
    """
      Fast Chebyshev transform.
    """
    if len(vals) == 1: return vals

    vals = vals[::-1] # re-ordering

    if endpoint:
      coef = jnp.fft.hfft(vals, axis=0, norm="forward")
      return coef[:len(vals)].at[1:-1].multiply(2)[:m]

    else:
      coef = jfft.dct(vals, axis=0) / len(vals)
      return coef.at[0].divide(2)[:m]

  @classmethod
  def nodal(cls, coef: Array, m: Maybe[int] = None, endpoint: bool = False) -> Array:
    """
      Inverse Chebyshev transform.
    """
    if len(coef) == 1: return coef

    if m is not None:
      shape = list(coef.shape)
      shape[0] = m
      coef = jnp.zeros(shape).at[:len(coef)].set(coef)

    if endpoint:
      coef = coef.at[1:-1].divide(2)

      full = jnp.concatenate([coef, coef[::-1][1:-1]])
      vals = jnp.fft.ihfft(full, axis=0, norm="forward").real

    else:
      coef = coef.at[0].multiply(2)
      vals = jfft.idct(coef * len(coef), axis=0)

    return vals[::-1] # re-ordering

  @classmethod
  def grad(cls, coef: Array) -> Array:

    # TODO: Fast Chebyshev differentiation
    grad = jnp.pad(differentiation(len(coef)), [(0, 1), (0, 0)])
    return jnp.tensordot(grad, coef, (1, 0))[..., jnp.newaxis]


def theta(n: int, endpoint: bool = False) -> Array:
  """
    Chebyshev nodes on the circle.
  """
  mode = None if endpoint else "cell"
  return utils.grid(n, mode=mode) * jnp.pi

def differentiation(n: int) -> Array:
  """
    Chebyshev differentiation matrix.
  """
  alt = jnp.pad(jnp.eye(2), [(0, n-3), (0, n-3)], mode="reflect").at[0].divide(2)
  coef = jnp.concatenate([jnp.zeros(n - 1)[:, jnp.newaxis], jnp.triu(alt)], axis=1)
  
  return coef * 4 * jnp.arange(n, dtype=float)
