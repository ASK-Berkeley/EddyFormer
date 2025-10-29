from nsm._typing import *
from numpy.polynomial import legendre

from ._base import Basis
class Legendre(Basis):

  """
    Shifted Legendre polynomials:
      - `(1 - x^2) Pn''(x) - 2 x Pn(x) + n (n + 1) Pn(x) = 0`
      - `Pn^~(x) = Pn(2 x - 1)`
  """

  @classmethod
  def grid(cls, m: int, endpoint: bool = False) -> Array:
    """
      Gauss-Legendre and Lobatto-Gauss-Legendre points.
    """
    return jnp.array(leggauss(m, endpoint)[0])[:, None]

  @classmethod
  @F.lru_cache
  def fn(cls, n: int) -> Callable:
    """
      Bonnet's recursion formula.
    """
    @jax.jit
    def call(xs: Array) -> Array:
      P = jnp.ones_like(xs), 2 * xs - 1

      for i in range(2, n):
        a, b = (i * 2 - 1) / i, (i - 1) / i
        P += a * P[-1] * P[1] - b * P[-2],

      return jnp.concatenate(P, axis=-1)

    return call

# --------------------------------- TRANSFORM -------------------------------- #

  @classmethod
  def modal(cls, vals: Array, m: Maybe[int] = None, endpoint: bool = False, use_fft: bool = False) -> Array:
    """
      Interpolate using Gauss-Legendre quadrature.
    """
    if not use_fft:
      fn = cls.fn(m:=m or len(vals))
      with jax.ensure_compile_time_eval():
        f = fn(cls.grid(len(vals), endpoint))
        f *= 2 * jnp.arange(m, dtype=float) + 1
        f = f.T * cls.quad(len(vals))()
      return jnp.tensordot(f, vals, axes=1)

    else:
      raise NotImplementedError

  @classmethod
  def nodal(cls, coef: Array, m: Maybe[int] = None, endpoint: bool = False, use_fft: bool = False) -> Array:
    """
      Matrix-vector multiplication.
    """
    if not use_fft: return cls.at(coef, cls.grid(m or len(coef), endpoint))
    else:
      raise NotImplementedError

  @classmethod
  def grad(cls, coef: Array) -> Array:
    raise NotImplementedError


def leggauss(m: int, lobatto: bool) -> Tuple[Array, Array]:
  """
    Gauss-Legendre points and quadrature weights on [0, 1].
  """
  if lobatto: m -= 1
  c = (0, ) * m + (1, )
  dc = legendre.legder(c)

  x = legendre.legroots(dc if lobatto else c)
  y = legendre.legval(x, c if lobatto else dc)

  if lobatto:
    x = np.concatenate([[-1], x, [1]])
    y = np.concatenate([[1], y, [1]])

  w = 1 / y ** 2
  if lobatto:
    w /= m * (m + 1)
  else:
    w /= 1 - x ** 2

  return (1 + x) / 2, w
