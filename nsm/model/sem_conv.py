from nsm.typing import *

class SEMConv(nn.Module):

  odim: Union[int, Shape]

  kernel_mode: Shape
  kernel_size: Shape
  use_bias: bool

  kernel_init_std: float = 1e-7
  bias_init: str = "zeros"

  def kernel_init(self, prng: Array, shape: Shape) -> Array:
    return jrand.normal(prng, shape) * self.kernel_init_std

  @nn.compact
  def __call__(self, ϕ: SEM) -> Array:
    """
    """
    idim, odim = ϕ.nodal.shape[-1], self.odim
    if isinstance(odim, Tuple): odim = math.prod(odim)

    xs = []
    for n, s in enumerate(self.kernel_size):

      w = self.param(f"W_{n}", self.kernel_init, (self.kernel_mode[n] * s, idim, odim))
      x = utils.reshape(sem_conv(ϕ, w, n, s), self.odim, -1)

      if self.use_bias:
        bias_init = getattr(nn.initializers, self.bias_init)
        x += self.param(f"b_{n}", bias_init, self.odim)

      xs.append(x)

    return xs

# ---------------------------------------------------------------------------- #
#                                SEM CONVOLUTION                               #
# ---------------------------------------------------------------------------- #

def _nuift(coef: Array, xs: Array) -> Array:
  """
    Evaluate the Fourier kernel.

    Args:
      coef: Coefficients of the Fourier kernel.
      xs: Coordinates of collocation points.
  """
  r, i = jnp.split(coef, (m:=(n:=len(coef)) // 2 + 1, ))
  i = jnp.zeros(r.shape).at[n-m:0:-1].set(i)

  xs *= 2 * jnp.pi * jnp.arange(m, dtype=float)
  f = jnp.exp(1j * xs.astype(complex)).at[..., 1:-1].mul(2)

  return jnp.tensordot(f.real, r, 1) - jnp.tensordot(f.imag, i, 1)

@jax.named_call
def sem_conv(ϕ: SEM, coef: Array, n: int, s: int, unroll: bool = False) -> Array:
  """
    Numerical quadrature-based convolution.
    
    Args:
      ϕ: Input feature fields on SEM.
      coef: Coefficients of the Fourier kernel.
      n: Axis to perform the convolution.
      s: Window size of the compact kernel.
      unroll: Unroll the loop over the window.
  """
  ns = "".join(map(str, range(ϕ.ndim)))
  ms = ns.replace(str(n), "m")
  
  #1. Gaussian quadrature
  
  with jax.ensure_compile_time_eval():
    eps = jnp.finfo(ϕ.nodal.dtype).eps
    w = jax.vmap(ϕ.T.quad(m:=ϕ.mode[n]))
    
    x0 = ϕ.T.grid(m).squeeze(1) - s / 2
    interval = jnp.stack([x0, x0 + s], axis=1)

    ws = []
    for i in [*range((r:=s - s // 2) + 1), *range(-r, 0)]:
      ws.append(w(jnp.clip(interval - i, -eps, 1 + eps)))
    ws = jnp.stack(ws, axis=0)
  
  eqn = f"{ns}...i, m{n}io, m{n} -> {ms}...o"
  def quad(fx: Array, i: Array) -> Array:
    """
    """
    x = ϕ.T.grid(m) + i.astype(float)
    w = ws[i] # w = lax.dynamic_index_in_dim(ws, i, keepdims=False)
    
    xy = ϕ.T.grid(m)[:, None] - x
    gxy = _nuift(coef, xy / s)
    
    if config.debug:
      logging.info(f"{eqn}: {fx.shape} {gxy.shape} {w.shape}")
    return jnp.einsum(eqn, fx, gxy, w)
  
  #2. sliding convolution window
    
  @F.partial(jax.jit, donate_argnames="out")
  def conv(i: Array, out: Array) -> Array:
    """
    """
    x = utils.reshape(ϕ.nodal, ϕ.mesh, ϕ.ndim, ϕ.ndim + 1)
    pad_r = jnp.take(x, jnp.r_[:r], axis:=ϕ.ndim + n)
    pad_l = jnp.take(x, jnp.r_[ϕ.mesh[n] - r:ϕ.mesh[n]], axis)

    f = jnp.concatenate([pad_l, x, pad_r], axis)
    fx = lax.dynamic_slice_in_dim(f, i + r, ϕ.mesh[n], axis)
    return out + quad(utils.reshape(fx, -1, ϕ.ndim, ϕ.ndim * 2), i)

  out = jnp.zeros_like(jax.eval_shape(quad, ϕ.nodal, 0))
  return lax.fori_loop(-r, 1 + r, conv, out, unroll=unroll)
