from nsm.typing import *
from configs.model import default

def get_config():
  cfg = default()
  cfg.file = __file__

  cfg.path = "eddyformer.EddyFormer"
  cfg.config = ConfigDict({
      "hdim": 32,
      "odim": 1,
      "depth": 10,
      "checkpoint": False,
      "basis": "cheb_elem",
      "mode": (25, 25),
      "mode_les": (5, 5),
      "mesh": (16, 16),
      "ffn_dim": 128,
      "activation": "gelu",
      # SEM CONVOLUTION
        "kernel_size": (2, 2),
        "kernel_size_les": (2, 2),
        "kernel_mode": (24, 24),
        "kernel_mode_les": (4, 4),
        "use_bias": False,
      # SEM ATTENTION
        "num_heads": 8,
        "heads_dim": 16,
        "window": (8, 8),
        "normalize": "KQ",
        "pos_encode": "KQ",
        "attn_impl": "standard",
        "precision": "default",
  })

  return cfg
