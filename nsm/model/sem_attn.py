from nsm.typing import *

@jax.named_call
def standard_attention(q: Array, k: Array, v: Array, precision: str,
                       mask: Maybe[Array] = None, aux_data: Maybe[PyTree] = None) -> Array:
  """
    Standard implementation of scaled dot product attention.

    Args:
      q/k/v: Shape (...mode, q/kv_length, num_heads, heads_dim).
      precision: Precision for attention weight calculation.
      mask: Optional dense mask applied on the weight matrix.
  """
  q /= jnp.sqrt(q.shape[-1])
  attn = jnp.einsum("...qhd, ...khd -> ...hqk", q, k, precision=precision)

  if aux_data is not None:
    for name, x in zip("qkv", (q, k, v)):
      aux_data[f"attn/{name}"].append(x.std())
    aux_data["attn"].append(attn.std())

  return jnp.einsum("...hqk, ...khd -> ...qhd",
    nn.softmax(attn, where=mask), v, precision=precision)
