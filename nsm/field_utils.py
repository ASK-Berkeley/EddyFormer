from nsm.typing import *

Field = List[Array]

def fft(xs: Field) -> Field:
  """
    Fourier transform of real functions. Using forward
    normalization in order to make interpolation easy.

    Args:
      xs: Field in the real space.
  """
  assert len(set(map(jnp.shape, xs))) == 1
  return [jnp.fft.fftn(x, norm="forward") for x in xs]

def ifft(xs_: Field) -> Field:
  """
    Inverse Fourier transform of real functions. The
    input is assumed to be symmetric for a real output.

    Args:
      xs_: Field in the Fourier space.
  """
  assert len(set(map(jnp.shape, xs_))) == 1
  return [jnp.fft.irfftn(x_, x_.shape, norm="forward") for x_ in xs_]

def wavenumber(*ns: int, complex = True) -> Field:
  """
    Wavenumbers in a 2*pi cube.

    Args:
      ns: Shape of the field.
      complex: Return complex k.
  """
  freq = jnp.array(1.0)
  if complex: freq *= 1j

  k = lambda n: jnp.fft.fftfreq(n).astype(freq.dtype) * freq * n
  return jnp.meshgrid(*map(k, ns), indexing="ij")

def index(*ns: int) -> Array:
  """
    Indices of modes in Fourier domain.

    Args:
      ns: Shape of the field.
  """
  return jnp.ix_(*[jnp.r_[-n//2+1:n//2+1] for n in ns])

def laplacian(*ns: int) -> Array:
  """
    Laplacian operator in Fourier domain.

    Args:
      ns: Shape of the field.
  """
  ks = wavenumber(*ns, complex=False)
  return -sum(k ** 2 for k in ks).astype(complex)

# ---------------------------------------------------------------------------- #
#                                  CONVERTION                                  #
# ---------------------------------------------------------------------------- #

def from_grid(u: Grid, has_t: bool = False) -> Field:
  assert u.ndim + 1 == u.value.ndim, "invalid vector field"

  idx = [slice(None, -1)] * u.ndim
  if has_t: idx[0] = slice(None)

  pbc = u.value[tuple(idx)]
  return list(jnp.moveaxis(pbc, -1, 0))

def to_grid(u: Field, size: Array, has_t: bool = False) -> Grid:

  pad_width = [(0, 1)] * len(shape(u))
  if has_t: pad_width[0] = (0, 0)

  value = map(F.partial(jnp.pad, pad_width=pad_width, mode="wrap"), u)
  return Grid(jnp.stack(list(value), axis=-1), size)

# ---------------------------------------------------------------------------- #
#                                 VECTOR FIELD                                 #
# ---------------------------------------------------------------------------- #

def shape(xs: Field) -> Shape:
  """
    Shape of the field.

    Args:
      xs: Field of interest.
  """
  ns, = set(map(jnp.shape, xs))
  return ns # must in same shape

def interp(xs: Field, ns: Shape, fourier = False) -> Field:
  """
    Fourier interpolation. For each dimension, truncate the
    modes if desired shape is larger, else pad with zeros.

    Args:
      xs: Field of interest.
      ns: Desired shape of the field.
      fourier: Spectral inputs/outputs.
  """
  assert len(shape(xs)) == len(ns)
  if shape(xs) == ns: return xs

  if fourier: xs_ = xs
  else: xs_ = fft(xs)

  ix = index(*[min(n, m) for n, m in zip(shape(xs), ns)])
  ys_ = [jnp.zeros(ns, dtype=x_.dtype).at[ix].set(x_[ix]) for x_ in xs_]

  if fourier: return ys_
  else: return ifft(ys_)

def gaussian(xs: Field, Δ: float, fourier = False) -> Field:
  """
    Gaussian filter of the field.

    Args:
      xs: Field of interest.
      Δ: Characteristic width of the filter.
      fourier: Spectral inputs/outputs.
  """
  if fourier: xs_ = xs
  else: xs_ = fft(xs)

  ns = shape(xs)
  ks = wavenumber(*ns, complex=False)

  filter = jnp.exp(-sum(k ** 2 for k in ks) * Δ ** 2 / 24)
  ys_ = [x_ * filter.astype(complex) for x_ in xs_]

  if fourier: return ys_
  else: return ifft(ys_)

def curl(xs: Field, fourier = False) -> Field:
  """
    Circulation of the field.

    Args:
      xs: Vector field of interest.
      fourier: Spectral inputs/outputs.
  """
  def cross(xs: Field, ys: Field) -> Field:
    """
      Cross product of vector fields.
    """
    return [xs[1] * ys[2] - xs[2] * ys[1],
            xs[2] * ys[0] - xs[0] * ys[2],
            xs[0] * ys[1] - xs[1] * ys[0]]

  if fourier: xs_ = xs
  else: xs_ = fft(xs)

  ns = shape(xs)
  ks = wavenumber(*ns)
  xs_ = cross(ks, xs_)

  if fourier: return xs_
  else: return ifft(xs_)

def vel2vor(vs: Field, fourier = False) -> Field:
  """
    Velocity to vorticity.

    Args:
      vs: Velocity field. For 2D inputs, returns
          the scalar vorticity on Z dimension.
      fourier: Spectral inputs/outputs.
  """
  if len(vs) == 2:
    assert len(shape(vs)) == 2

    vs = [v[:, :, None] for v in vs]
    vz = [jnp.zeros(shape(vs))]
    wz = vel2vor(vs + vz, fourier)[-1]
    return [wz.squeeze(-1)]

  if len(vs) == 3:
    assert len(shape(vs)) == 3
    return curl(vs, fourier)

  raise ValueError(f"invalid dimension {len(vs)}")

def vor2vel(ws: Field, fourier = False) -> Field:
  """
    Vorticity to velocity.

    Args:
      ws: Vorticity field. For 1D input, returns
          the velocity field on X-Y dimensions.
      fourier: Spectral inputs/outputs.
  """
  if len(ws) == 1:
    assert len(shape(ws)) == 2

    wz = ws[0][:, :, None]
    wxy = [jnp.zeros(wz.shape)] * 2
    vs = vor2vel(wxy + [wz], fourier)
    return [v.squeeze(-1) for v in vs[:2]]

  if len(ws) == 3:
    assert len(shape(ws)) == 3

    if fourier: ws_ = ws
    else: ws_ = fft(ws)

    lap = laplacian(*shape(ws)).at[0, 0, 0].set(1.0)
    vs_ = [-w_ / lap for w_ in curl(ws_, True)]

    if fourier: return vs_
    else: return ifft(vs_)

  raise ValueError(f"invalid dimension {len(ws)}")

# ---------------------------------------------------------------------------- #
#                                  STATISTICS                                  #
# ---------------------------------------------------------------------------- #

def energy_spectrum(vs: Field) -> Array:
  """
    Energy spectrum of the velocity field.

    Args:
      vs: Velocity field.
  """
  vs_ = fft(vs)
  v2 = sum(v_ * v_.conj() for v_ in vs_).real

  ks = wavenumber(*shape(vs), complex=False)
  k = jnp.round(jnp.sqrt(sum(k ** 2 for k in ks))).astype(int)
  # k = jnp.round(sum(map(jnp.abs, ks))).astype(int)

  Ek = jnp.zeros(jnp.max(k) + 1)
  return Ek.at[k].add(v2)

def two_point_structure(vs: Field, order: int) -> Array:
  """
    Two-point structure function.
  """
  ns = shape(vs)

  def moment(shift: Tuple[int]) -> Array:
    e = jnp.stack(shift) / jnp.maximum(jnp.linalg.norm(jnp.stack(shift)), 1)
    dv = jnp.stack([v - jnp.roll(v, shift, axis=range(v.ndim)) for v in vs], axis=-1)
    return jnp.mean((dv @ e) ** order)

  S = lax.map(moment, jnp.unravel_index(jnp.arange(math.prod(ns)), ns))
  r = jnp.round(jnp.sqrt(sum(x ** 2 for x in jnp.meshgrid(*map(jnp.arange, ns))))).astype(int).ravel()

  n = jnp.zeros(362).at[r].add(1)
  return jnp.zeros(362).at[r].add(jnp.array(S)) / n

def statistics(vs: Field, nu: float) -> Dict[str, Array]:
  """
    Statistics of the turbulent flow.

    Args:
      vs: Velocity field.
      nu: Viscosity coefficient.
  """
  vs_ = fft(vs)

  ks = wavenumber(*shape(vs))
  dvs_ = [v_ * k for v_ in fft(vs) for k in ks]

  E = sum(jnp.sum(v_ * v_.conj()).real for v_ in vs_)
  eps = 2 * nu * sum(jnp.sum(dv_ * dv_.conj()).real for dv_ in dvs_)

  rms = jnp.sqrt(2 / 3 * E)
  tms = jnp.sqrt(15 * nu / eps) * rms

  Re = rms * tms / nu

  return {
    "E": E,
    "Re": Re,
    "eps": eps,
    "rms": rms,
    "tms": tms,
  }
