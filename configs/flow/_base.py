from nsm.typing import *
from nsm.flow import Flow

from importlib import import_module

def default() -> ConfigDict:

  return ConfigDict({
    "file": __file__,
    "path": placeholder(str),
    "config": placeholder(ConfigDict),
  })

def resolve(cfg: ConfigDict) -> Flow:
  cfg = cfg.copy_and_resolve_references()

  mod, cls = cfg.path.rsplit(".", 1)
  mod = import_module(f"nsm.flow.{mod}")

  return getattr(mod, cls)(**cfg.config)
