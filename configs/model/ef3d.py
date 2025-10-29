from nsm.typing import *
from configs.model import default

def get_config():
  cfg = default()
  cfg.file = __file__

  cfg.path = "eddyformer.EddyFormer"
  cfg.config = ConfigDict({
      "hdim": 32,
      "odim": 3,
      "depth": 4,
      "checkpoint": False,
      "basis": "cheb_elem",
      "mode": (13, 13, 13),
      "mode_les": (6, 6, 6),
      "mesh": (8, 8, 8),
      "ffn_dim": 128,
      "activation": "gelu",
      # SEM CONVOLUTION
        "kernel_size": (2, 2, 2),
        "kernel_size_les": (2, 2, 2),
        "kernel_mode": (12, 12, 12),
        "kernel_mode_les": (5, 5, 5),
        "use_bias": False,
      # SEM ATTENTION
        "num_heads": 4,
        "heads_dim": 32,
        "window": placeholder(tuple),
        "normalize": "KQ",
        "pos_encode": "KQ",
        "attn_impl": "standard",
        "precision": "default",
  })

  return cfg
