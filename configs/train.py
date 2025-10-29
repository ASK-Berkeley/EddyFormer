from nsm.typing import *

from importlib import import_module
from ml_collections.config_dict import placeholder

def get_config() -> ConfigDict:

  return ConfigDict({
    "file": __file__,
    "path": "train.Trainer",
    # random seed; default to current time
    "seed": placeholder(int),
    # compile train and eval iterations
    "compile": True,
# ---------------------------------------------------------------------------- #
#                                     TRAIN                                    #
# ---------------------------------------------------------------------------- #
    "train": {
      # total epochs
      "epoch": 10000,
      # total iterations
      "iteration": placeholder(int),
      # optimizer learning rate
      "learning_rate": 1e-3,
      # learning rate scheduler
      "scheduler": placeholder(str),
      # optimizer name
      "optimizer": "adam",
# ----------------------------------- BATCH ---------------------------------- #
      # training batch size
      "batch_size": 16,
      # vectorization
      "vmap_batch": 1,
      # sharded over GPUs
      "batch_sharding": True,
      # number of CPU workers
      "num_cpu_proc": -1,
# ----------------------------------- LOSS ----------------------------------- #
      # rollout steps
      "window": 1,
      # gradient clipping
      "gradient_clip": placeholder(float),
    },
# ---------------------------------------------------------------------------- #
#                                      LOG                                     #
# ---------------------------------------------------------------------------- #
    "log": {
      # write tensorboard event
      "use_tensorboard": True,
      # load path; default to no loading
      "load_path": placeholder(str),
      # load iteration if `load_path` is not None
      "load_ckpt_it": placeholder(int),
      # save path; default to a timestamp
      "save_path": placeholder(str),
      # evaluation period
      "test_period": 100,
      # checkpoint interval
      "ckpt_period": 6 * 3600,
    },
  })

def resolve(cfg: ConfigDict):
  cfg = cfg.copy_and_resolve_references()

  if cfg.seed is None:
    cfg.seed = int(time.time())

  def make_optim(cfg: ConfigDict):

    if cfg.scheduler is None:
      scheduler = float(cfg.learning_rate)

    if cfg.scheduler == "warmup":
      scheduler = optax.schedules.linear_schedule(0, cfg.learning_rate, 100)

    optimizer = getattr(optax, cfg.optimizer)

    with cfg.ignore_type():
      cfg.optimizer = optimizer(scheduler)
      cfg.scheduler = scheduler

  make_optim(cfg.train)
  if "pretrain" in cfg:
    make_optim(cfg.pretrain)

  src, cls = cfg.path.rsplit(".", 1)
  mod = import_module(f"nsm.{src}")

  return getattr(mod, cls).make_train(cfg)
