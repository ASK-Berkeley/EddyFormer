from nsm.typing import *

# ---------------------------------------------------------------------------- #
#                                 TIME-STEPPING                                #
# ---------------------------------------------------------------------------- #

class Step(Protocol):

  def __call__(self, x: PyTree, dt: float) -> PyTree:
    """
      An iterative numerical solver.

      Args:
        ws:  Initial scalar or vector vorticity.
        ts: Recording time step(s).
    """

# Butcher tableau of some classic RK schemes. Omitting the time
# variable since the dynamics are assumed to be time-independent.

RK = {
  "rk2": [[1], [1/2, 1/2]],
  "rk2-ralston": [[2/3], [1/4, 3/4]],
  "rk4": [[1/2], [0, 1/2], [0, 0, 1],[1/6, 1/3, 1/3, 1/6]],
  "rk4-3/8": [[1/3], [-1/3, 1], [1, -1, 1], [1/8, 3/8, 3/8, 1/8]],
}

def explicit(x: PyTree, f: PyTree, dt: float) -> PyTree:
  return jax.tree.map(lambda xi, fi: xi + fi * dt, x, f)

def make_step(scheme: str, f: Callable[[PyTree], PyTree],
     implicit: Callable[[PyTree, PyTree, float], PyTree] = explicit) -> Step:
  """
    Generic time stepping using a Runge-Kutta scheme.

    Args:
      scheme: Explicit time stepping scheme. One of ["rk2", "rk3", "rk4"].
      f: The dynamics of the system. It is assumed to be time-independent.
      implicit: Implicit stepping in the inner loop. Default to explicit.
  """
  def step(x: PyTree, dt: float) -> PyTree:

    def weighed_sum(xs: List[PyTree], ys: List[float]) -> PyTree:
      """Weighed sum of a list of pytrees with an identical structure."""
      return jax.tree.map(lambda *xsi: sum(xi * yi for xi, yi in zip(xsi, ys)), *xs)

    x_n = x
    fs = []

    for a in RK[scheme]:
      fs.append(f(x_n))

      f_n = weighed_sum(fs, a)
      x_n = implicit(x, f_n, dt)

    return x_n

  return step

# ---------------------------------------------------------------------------- #
#                                  WHILE LOOP                                  #
# ---------------------------------------------------------------------------- #

def while_loop(cond: Callable[[PyTree], bool],
               body: Callable[[PyTree], PyTree],
               init: PyTree, max_steps: int):
  """
    A memory-efficient while-loop.

    Args:
      cond: Condition (carry -> bool).
      body: Loop body (carry -> carry).
      max_steps: Maximum number of iterations.
  """
  ckpt = math.ceil(math.sqrt(max_steps))

  @jax.checkpoint
  def outer(_, carry: PyTree) -> PyTree:
    body_wrap = lambda _, x: lax.cond(cond(x), body, lambda x: x, x)
    return lax.cond(cond(carry), lambda: lax.fori_loop(0, ckpt, body_wrap, carry), lambda: carry)

  return lax.fori_loop(0, ckpt, outer, init)
