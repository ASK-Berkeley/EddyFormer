from nsm.typing import *

from nsm.flow import Flow
from nsm.model import Model

class BasicBlock(nn.Module):

  odim: int
  zero_init: bool

  @property
  def conv(self):
    return F.partial(
      nn.Conv,
      kernel_size=(3, 3),
      padding="CIRCULAR",
      use_bias=False,
    )

  @nn.compact
  def __call__(self, x: Array) -> Array:
    identity = x

    init = None
    if self.zero_init:
      init = nn.zeros_init()

    out = nn.relu(self.conv(x.shape[-1])(x))
    out = self.conv(self.odim, kernel_init=init)(out)

    if identity.shape[-1] != self.odim:
      identity = nn.Dense(self.odim, use_bias=False)(identity)

    out += identity
    return nn.relu(out)

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

class ResNet(nn.Module, Model):

  odim: int
  layers: List[int]
  zero_init: bool

  def layer(self, x: Array, hdim: int, blocks: int) -> Array:
    x = BasicBlock(hdim, self.zero_init)(x)
    for _ in range(1, blocks):
      x = BasicBlock(hdim, self.zero_init)(x)
    return x

  @nn.compact
  def __call__(self, flow: Flow, ic: Grid, out: Maybe[Grid] = None, return_aux: bool = False) -> Model.Output:
    """
    """
    aux = dd(lambda: []) if return_aux else None
    ϕ, u0 = flow.process(ic, Grid(jnp.zeros(ic.resolution), ic.size), aux)

    x = jnp.concatenate([ϕ.coords, ϕ.value], axis=-1)
    x = nn.Conv(64, (7, 7), padding="CIRCULAR", use_bias=False)(x)
    x = nn.relu(x)

    x = self.layer(x, 64, self.layers[0])
    x = self.layer(x, 128, self.layers[1])
    x = self.layer(x, 256, self.layers[2])

    u = Grid(nn.Dense(self.odim)(x), ϕ.size)

    u = flow.project(u, u0, out or ic)
    return Model.Output(u, dict(aux or {}))
