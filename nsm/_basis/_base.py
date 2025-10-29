from nsm._typing import *
from quadax import quadgk

class Basis(Protocol):

  """
    Spectral basis for function approximation. It is
    assumed to be defined on a unit interval `[0, 1]`.
  """

  @classmethod
  def fn(cls, n: int) -> Callable[[Array], Array]:
    """
      Function values at given points.

      Args:
        n: Order of the basis.
    """

  @classmethod
  def at(cls, coef: Array, xs: Array) -> Array:
    """
      Evaluate the basis at given points.

      Args:
        coef: Coefficients of the basis.
        xs: Evaluated points.
    """
    f = cls.fn(len(coef))(xs)
    return jnp.tensordot(f, coef, axes=1)

# -------------------------------- QUADRATURE -------------------------------- #

  @classmethod
  def grid(cls, m: int) -> Array:
    """
      Collocation points.

      Args:
        m: Number of modes.
    """

  @classmethod
  @F.lru_cache
  def quad(cls, m: int, **kwargs) -> Array:
    """
      Integration formulas of interpolatory type.

      Args:
        m: Number of modes.
        kwargs: See `scipy.integrate.quad`.
    """
    if "epsabs" not in kwargs:
      kwargs["epsabs"] = 1e-7

    @jax.jit
    def quad(interval: Maybe[Array] = None) -> Array:
      """
        Calculate numerical quadrature. The interval
        is assumed to be in the range of [0, 1].

        Args:
          interval: Left/right boundary.
      """
      if interval is None:
        interval = jnp.array([0, 1])
      assert interval.ndim == 1, \
        f"invalid {interval.ndim = }"

      if config.debug:
        msg = f"Calculating {m}-point {cls.__qualname__} quadrature"
        logging.info(f"{msg} with {kwargs}..." if kwargs else f"{msg}...")

      with jax.numpy_dtype_promotion("standard"): # quadax is not strict
        quad = quadgk(lambda x: cls.fn(m)(x[None]), interval, **kwargs)[0]

      return jnp.linalg.solve(cls.fn(m)(cls.grid(m)).T, quad)

    return quad

# --------------------------------- TRANSFORM -------------------------------- #

  @classmethod
  def modal(cls, vals: Array) -> Array:
    """
      Interpolate values on collocation points.

      Args:
        vals: Basis values on the grids.
    """

  @classmethod
  def nodal(cls, coef: Array) -> Array:
    """
      Evaluate values on collocation poitns.

      Args:
        coef: Coefficients of the basis.
    """

  @classmethod
  def grad(cls, coef: Array) -> Array:
    """
      Take derivative on the coefficients.

      Args:
        coef: Coefficients of the basis.
    """
