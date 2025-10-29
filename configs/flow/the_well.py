from nsm.typing import *
from configs.flow import default

def get_config():
  cfg = default()
  cfg.file = __file__

  cfg.path = "the_well.TheWell"
  cfg.config = ConfigDict({
    "base_dir": placeholder(str),
    "name": placeholder(str),
  })

  return cfg
