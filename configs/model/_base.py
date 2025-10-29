from nsm.typing import *
from nsm.model import Model

from importlib import import_module
from ml_collections.config_dict import placeholder

def default() -> ConfigDict:

  return ConfigDict({
    "file": __file__,
    "path": placeholder(str),
    "config": placeholder(ConfigDict),
  })

def resolve(cfg: ConfigDict) -> Model:
  cfg = cfg.copy_and_resolve_references()

  src, cls = cfg.path.rsplit(".", 1)
  mod = import_module(f"nsm.model.{src}")

  return getattr(mod, cls)(**cfg.config)
