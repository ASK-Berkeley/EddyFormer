from nsm._typing import *
from ._base import Basis

@F.cache
def make_element(T: Type[Basis]) -> Type[Basis]:
  """
    Wrap a polynomial into SEM basis.

    Args:
      T: Base polynomial basis class.
  """
  class C0(T):

    """
      Boundary-interior decomposition.
    """

    @classmethod
    def grid(cls, m: int) -> Array:
      return T.grid(m, endpoint=True)

    @classmethod
    def at(cls, coef: Array, xs: Array) -> Array:
      x = xs.squeeze(-1)
      x = jnp.expand_dims(x, range(x.ndim, x.ndim + coef.ndim - 1))

      l, r, coef_i = jnp.split(coef, [1, 2])
      vals = x * (1 - x) * T.at(coef_i, xs)
      return l.squeeze(0) * (1 - x) + r.squeeze(0) * x + vals

# --------------------------------- TRANSFORM -------------------------------- #

    @classmethod
    def modal(cls, vals: Array, m: Maybe[int] = None) -> Array:
      """
      """
      x = cls.grid(len(vals)).squeeze(-1)
      x = jnp.expand_dims(x, range(1, vals.ndim))

      with jax.ensure_compile_time_eval():
        f = T.fn(n:=m or len(vals))(x0:=T.grid(n))
        i = jnp.linalg.inv(T.modal(f * x0 * (1 - x0)))[:n - 2]

      coef = T.modal(vals - (1 - x) * vals[0] - x * vals[-1], n, endpoint=True)
      return jnp.concatenate([vals[[0, -1], ...], jnp.tensordot(i, coef, 1)], axis=0)

    @classmethod
    def nodal(cls, coef: Array, m: Maybe[int] = None) -> Array:
      """
      """
      x = cls.grid(m or len(coef)).squeeze(-1)
      x = jnp.expand_dims(x, range(1, coef.ndim))

      l, r, coef_i = jnp.split(coef, [1, 2], axis=0)
      vals = x * (1 - x) * T.nodal(coef_i, len(x), endpoint=True)
      return l * (1 - x) + r * x + vals

  C0.__qualname__ = f"{T.__qualname__}_SEM"
  return C0
