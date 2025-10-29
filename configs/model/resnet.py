from nsm.typing import *
from configs.model import default

def get_config():
  cfg = default()
  cfg.file = __file__

  cfg.path = "cnn.ResNet"
  cfg.config = ConfigDict({
      "odim": 1,
      "layers": (2, 2, 2),
      "zero_init": True,
  })

  return cfg
