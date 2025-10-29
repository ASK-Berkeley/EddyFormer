from nsm.typing import *
from configs.model import default

def get_config():
  cfg = default()
  cfg.file = __file__

  cfg.path = "fno.FFNO"
  cfg.config = ConfigDict({
      "hdim": 32,
      "odim": 3,
      "depth": 24,
      "mode": (64, 64, 64),
      "activation": "gelu",
  })

  return cfg
