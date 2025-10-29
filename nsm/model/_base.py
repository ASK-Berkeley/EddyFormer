from nsm.typing import *
from nsm.flow import Flow

class Model(Protocol):

  @struct.dataclass
  class Output:

    u: DataType
    aux: Maybe[PyTree]

  def dtype(self, ic: Grid) -> DataType:
    """
      Datatype of the model.

      Args:
        ic: Initial condition.
    """
    return ic # placeholder

  def __call__(self, flow: Flow, ic: Grid, out: Maybe[Grid] = None, return_aux: bool = False) -> Output:
    """
      Run the model to simulate a flow.

      Args:
        flow: Flow instance.
        ic: Initial condition.
        out: Target datatype.
        return_aux: return auxiliary data.
    """
    aux = dd(lambda: []) if return_aux else None
    ϕ, u0 = flow.process(ic, self.dtype(ic), aux)

    u = self.forward(ϕ, aux)
    u = flow.project(u, u0, out or ic)

    return Model.Output(u, dict(aux or {}))

  def forward(self, ϕ: DataType, aux_data: Maybe[PyTree]) -> DataType:
    """
      Forward pass of the model.

      Args:
        ϕ: Processed initial condition.
        aux_data: Mutable auxiliary data.
    """
