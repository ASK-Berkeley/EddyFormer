from nsm.typing import *
from nsm import field_utils

from nsm.field_utils import Field
from nsm.solver import make_step, while_loop

class Forcing(Protocol):

  def __call__(self, us_: Field) -> Field:
    """
      Forcing term in the Fourier form.

      Args:
        us_: Current field.
    """

class Config(Protocol):

  @property
  def courant(self) -> Maybe[float]:
    """
      Maximum Courant number.
    """

  @property
  def dt_scheme(self) -> str:
    """
      Explicit time stepping.
    """

  @property
  def dt(self) -> Maybe[float]:
    """
      Time stepping inverval:
        - If given, then using a fixed time stepping.
        - else using dt determined by CFL conditions.
    """

  @property
  def max_steps(self) -> Maybe[float]:
    """
      Maximum number of time steps.
    """

  @property
  def dealias(self) -> Maybe[str]:
    """
      Dealiasing scheme name.
    """

# ---------------------------------------------------------------------------- #
#                                 TIME STEPPING                                #
# ---------------------------------------------------------------------------- #

def make_dt(ns: Shape, courant: float, form: str, fourier = False):
  """
    Stable time stepping based on CFL condition.

    Args:
      ns: Shape of the field.
      courant: Maximum Courant number.
      form: Velocity or vorticity input.
      fourier: Spectral inputs/outputs.
  """
  dx = 2 * jnp.pi / max(ns)

  def cfl(u: Field) -> Array:

    if form == "velocity": vs = u
    else: vs = field_utils.vor2vel(u, fourier)

    if fourier: vs = field_utils.ifft(vs)
    vmax = sum(jnp.max(jnp.abs(v)) for v in vs)

    return courant * dx / vmax

  return cfl

def make_filter(ns: Shape, scheme: str) -> Maybe[Array]:
  """
    Filter-based dealiasing schemes.

    Args:
      ns: Shape of the field.
      rule: One of ["2/3", "smooth"].
  """
  if scheme == "2/3":

    ix = field_utils.index(*[max(1, n*2//3) for n in ns])
    return jnp.zeros(ns).at[ix].set(1.0).astype(complex)

  if scheme == "smooth":
    
    ks = field_utils.wavenumber(*ns, complex=False)

    kn = jnp.stack([k / (n//2+1) for k, n in zip(ks, ns)])
    return jnp.exp(-36*jnp.max(jnp.abs(kn), axis=0)**36).astype(complex)

  raise ValueError(f"invalid filter {scheme}")

# ---------------------------------------------------------------------------- #
#                                PSEUDO SPECTRAL                               #
# ---------------------------------------------------------------------------- #

@struct.dataclass
class State:

  t: Array    # remaining time
  dt: Array   # next step size
  us_: Field  # current field
  aux: PyTree = None # auxiliary data

@dataclass
class PseudoSpectral:

  """
    Pseudo-spectral solver for homogeneous turbulence on periodic domains. Using
    a combination of implicit and explicit time stepping schemes in the spectral
    domain. The solution is recorded at given times by stable time stepping.
  """

  cfg: Config
  f: Forcing

  form: str
  nu: float
  s: float

  def __init__(self, cfg: Config, f: Forcing, form: str, nu: float, s: int = 1):
    self.cfg, self.f, self.form, self.nu, self.s = cfg, f, form, nu, s

  def solve3d(self, ic: Field, ts: Union[float, Array], aux_data: PyTree) -> Field:
    """
      Solve the 3D NS equation. The given recording time
      is used to return the solution at those points.
    """
    ns = field_utils.shape(ic)
    ks = field_utils.wavenumber(*ns)

    lap = field_utils.laplacian(*ns)
    ilap = 1 / lap.at[0,0,0].set(1.0)

    filter = make_filter(ns, self.cfg.dealias)
    def nonlinear(us_: Field) -> Field:
      """
        Advection and forcing.
      """
      if self.form == "velocity":
        vs = field_utils.ifft(us_)

      if self.form == "vorticity":
        vs_ = field_utils.vor2vel(us_, True)
        vs = field_utils.ifft(vs_)

      def advect(u_: Array) -> Array:
        """
          The advection term `v·∇u`. Here `u`
          can be either velocity or vorticity.
        """
        dus_ = [u_ * k for k in ks]
        dus = field_utils.ifft(dus_)

        vdu = sum([v * du for v, du in zip(vs, dus)])
        vdu_, = field_utils.fft([vdu])

        return vdu_ * filter

      return [f_ - advect(u_) # / self.s
          for f_, u_ in zip(self.f(us_), us_)]

    def project(us_: Field) -> Field:
      """
        Project the solution field.
      """
      us_ = [u_.at[0, 0, 0].set(0.0) for u_ in us_]

      if self.form == "velocity": # div-free
        div = sum(k * u_ for k, u_ in zip(ks, us_))
        us_ = [u_ - k * div * ilap for k, u_ in zip(ks, us_)]

      return us_

    def crank_nicolson(us_: Field, fs_: Field, dt: float) -> Field:
      """
        Implicit step that solves the next state.
      """
      if isinstance(dt, Array): dt = dt.astype(complex)

      def solve(u_: Array, f_: Array) -> Array:
        """
          Crank-Nicolson step.
            0 | 0
            1 | 1/2 1/2
            -----------
              | 1/2 1/2
        """
        nu = self.nu / self.s ** 2
        u_ += dt * (f_ + nu * lap * u_ / 2)
        return u_ / (1 - nu * lap * dt / 2)

      return [solve(u_, f_) for u_, f_ in zip(us_, fs_)]

    cfl = make_dt(ns, self.cfg.courant, form=self.form, fourier=True)
    step_dt = make_step(self.cfg.dt_scheme, nonlinear, crank_nicolson)

    def step(us_aux: Tuple[Field, PyTree], Δt: float) -> Tuple[Tuple[Field, PyTree], Field]:
      """
        Explicit-implicit time stepping.
      """
      us_, aux = us_aux
      if aux is not None: aux["num_steps"] = aux["cfl"] = 0.
      init = State(Δt, self.cfg.dt or cfl(us_), us_, aux)

      def cond(state: State) -> bool:
        return jnp.logical_and(state.t > state.dt, state.dt > 0)

      def body(state: State) -> State:
        us_ = project(step_dt(state.us_, state.dt))
        if state.aux is not None:
          state.aux["num_steps"] = state.aux["num_steps"] + 1
          state.aux["cfl"] = jnp.maximum(state.aux["cfl"], state.dt / cfl(state.us_))
        return State(state.t - state.dt, self.cfg.dt or cfl(us_), us_, state.aux)

      n = self.cfg.max_steps
      if self.cfg.dt is not None and isinstance(Δt, float):
        n = self.cfg.max_steps or int(Δt // self.cfg.dt)

      if not n: last = lax.while_loop(cond, body, init)
      else: last = while_loop(cond, body, init, max_steps=n)

      state = body(State(last.t, last.t, last.us_, last.aux))
      return (state.us_, last.aux), field_utils.ifft(state.us_)

    us_ = field_utils.fft(ic)

    if isinstance(ts, float) or ts.ndim == 0:
      (_, aux), us = step((us_, aux_data), ts)
      if aux_data:
        aux_data.update(aux)
      return us

    if ts.ndim == 1:
      ts = ts[ord:=jnp.argsort(ts)]
      i_ord = jnp.argsort(ord)

      (us_, aux), us = step((us_, aux_data), ts[0])
      (_, aux), uss = lax.scan(step, (us_, aux), ts[1:] - ts[:-1])
      if aux_data: aux_data.update(aux)
      return [jnp.concatenate([u[None], us], 0)[i_ord] for u, us in zip(us, uss)]    

    raise ValueError(f"invalid dimension {ts.ndim}")

  def solve2d(self, ic: Field, ts: Union[float, Array], aux_data: PyTree) -> Field:
    """
      Solve the 2D NS equation. The initial condition is either
      a scalar vorticity or vector velocity field. It's converted
      into 3D by padding an empty `z` component.
    """
    ic = [u[:, :, None] for u in ic]

    if len(ic) == 2 and self.form == "velocity":

      ic += [jnp.zeros(field_utils.shape(ic))]
      return [v.squeeze(-1) for v in self.solve3d(ic, ts, aux_data)[:2]]

    if len(ic) == 1 and self.form == "vorticity":

      ws = [jnp.zeros(field_utils.shape(ic))] * 2
      return [self.solve3d(ws + ic, ts, aux_data)[-1].squeeze(-1)]

    raise ValueError(f"invalid dimension {len(ic)} in {self.form} form")

  def __call__(self, ic: Field, ts: Union[float, Array], aux_data: PyTree = None) -> Field:
    """
      Solve the NS equation on periodic grids. The input grid is
      interpolated in the Fourier space and solved using pseudo-
      spectral methods.
    """
    ns = field_utils.shape(ic)
    if len(ns) == 2:
      return self.solve2d(ic, ts, aux_data)
    if len(ns) == 3:
      assert self.form == "velocity"
      return self.solve3d(ic, ts, aux_data)
