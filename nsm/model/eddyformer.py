from nsm.typing import *
from ._base import Model

from . import sem_attn
from .sem_conv import SEMConv

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

class EddyFormer(nn.Module, Model):

  hdim: int
  odim: int
  depth: int
  checkpoint: bool

  ffn_dim: int
  activation: str

  # SEM CONVOLUTION

  kernel_size: Shape
  kernel_size_les: Shape
  kernel_mode: Shape
  kernel_mode_les: Shape
  use_bias: bool

  # SEM ATTENTION

  num_heads: int
  heads_dim: int
  window: Maybe[Shape]

  normalize: str
  pos_encode: str

  attn_impl: str
  precision: str

  # BASE

  basis: str
  mode: Shape

  mesh: Maybe[Shape] = None
  mode_les: Maybe[Shape] = None

  def dtype(self, ic: Grid) -> SEM: # placeholder type
    return SEM(self.basis, ic.size, self.mesh, self.mode)

  @property
  def conv(self):
    """
      SEM-based convolution.
    """
    return F.partial(
      SEMConv,
      odim=self.hdim,
      kernel_mode=self.kernel_mode,
      kernel_size=self.kernel_size,
      use_bias=self.use_bias,
    )

  @property
  def conv_les(self):
    """
      SEMConv in the LES stream.
    """
    return F.partial(
      self.conv,
      kernel_mode=self.kernel_mode_les,
      kernel_size=self.kernel_size_les,
    )

  def ffn(self, x: Array, odim: Maybe[int] = None) -> Array:
    """
      Feed-forward network.
    """
    x = getattr(nn, self.activation)(nn.Dense(self.ffn_dim)(x))
    return nn.DenseGeneral(odim or self.hdim)(x)

  def project(self, ψ: SEM, name: str, layer: int) -> Array:
    """
      Attention (query, key, value) projection.

      Args:
        ψ: LES feature on SEM.
        name: Name of the head.
        layer: Index of current layer.
    """
    ys = self.conv_les(
      name=f"SEMConv_{name}{layer}",
      odim=(self.num_heads, self.heads_dim),
      use_bias=self.use_bias or name in self.pos_encode,
    )(ψ)

    for n in range(ψ.ndim):

      if name in self.normalize:
        ys[n] = nn.LayerNorm(
          name=f"LayerNorm_{name}{layer}_{n}",
          feature_axes=(*range(ψ.ndim), -1),
          reduction_axes=(*range(ψ.ndim), -1),
          use_bias=False,
        )(ys[n])

      if name in self.pos_encode:
        f, g = jnp.split(ys[n], 2, axis=-1)
        f += 1 # self.param(f"pe_{name}{layer}_{n}", nn.initializers.zeros, f.shape[-1])
        x = ψ.coords[..., None, [n]] * jnp.arange(self.heads_dim // 2, dtype=float)
        ys[n] = jnp.concatenate([jnp.cos(x) * f - jnp.sin(x) * g,
                                 jnp.sin(x) * f + jnp.cos(x) * g], axis=-1)

    return jnp.concatenate(ys, axis=-1)

  def layer(self, ϕ: SEM, n: int, les: Array, sgs: Array, aux_data: Maybe[PyTree]) -> Tuple[Array, Array]:
    """
      EddyFormer layer.

      Args:
        ϕ: SEM datatype.
        n: Index of current layer.
        les: Input LES hidden feature.
        sgs: Input SGS hidden feature.
        aux_data: Mutable auxiliary data.
    """
    try: attn_impl = getattr(sem_attn, f"{self.attn_impl}_attention")
    except AttributeError: raise ValueError(f"`{self.attn_impl}` attention not found")

    mask = None
    if self.window:
      masks = []
      for coords, l, s in zip(jnp.unravel_index(jnp.arange(len(ϕ)), ϕ.mesh), ϕ.mesh, self.window):
        masks.append(jnp.minimum(dis:=jnp.abs(coords[:, None] - coords[None, :]), l - dis) < s)
      mask = F.reduce(jnp.logical_and, masks)

    # shape of qkv: (...ϕ.mode, mesh, num_heads, ϕ.ndim * heads_dim)
    q, k, v = (self.project(ϕ.new(nodal=les), name, n) for name in "QKV")
    attn = attn_impl(q, k, v, precision=self.precision, mask=mask, aux_data=aux_data)

    les += nn.DenseGeneral(self.hdim, axis=(-2, -1))(attn)
    les += self.ffn(sum(self.conv_les()(ϕ.new(nodal=les))))

    eps = self.param(f"scale_les_{n}", nn.initializers.constant(1e-7), self.hdim)
    sgs += ϕ.new(nodal=les).to(self.mode).nodal * eps
    sgs += self.ffn(sum(self.conv()(ϕ.new(nodal=sgs))))

    if aux_data is not None:
      aux_data[f"scale/les_{n}"] = eps.std()

    return les, sgs

# ---------------------------------- FORWARD --------------------------------- #

  @nn.compact
  def forward(self, ϕ: SEM, aux_data: Maybe[PyTree]) -> SEM:
    """
      Forward pass of EddyFormer.
    """
    coords = jnp.repeat(ϕ.grid[..., jnp.newaxis, :], len(ϕ), axis=ϕ.ndim)
    sgs = nn.Dense(self.hdim)(jnp.concatenate([ϕ.nodal, coords], axis=-1))

    sgs += self.ffn(sum(self.conv()(ϕ.new(nodal=sgs))))
    les = ϕ.new(nodal=sgs).to(self.mode_les).nodal

    layer = EddyFormer.layer
    if self.checkpoint:
      layer = nn.checkpoint(layer, static_argnums=(0, 1, 2))

    for n in range(self.depth):
      les, sgs = layer(self, ϕ, n, les, sgs, aux_data)

    u_les = self.ffn(les, odim=self.odim)
    u_sgs = self.ffn(sgs, odim=self.odim)

    eps = self.param("scale_sgs", nn.initializers.constant(1e-7), self.odim)
    u = ϕ.new(nodal=u_les).to(self.mode).nodal + u_sgs * eps

    if aux_data is not None:
      aux_data["scale/sgs"] = list(eps)

    return ϕ.new(nodal=u)
