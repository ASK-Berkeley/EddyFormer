from typing import *
from absl import logging

import matplotlib.pyplot as plt, scienceplots
plt.style.use(["science", "grid", "no-latex"])

import os
import re
import csv
import datetime
import pickle
import shutil

from ml_collections import ConfigDict
from collections import defaultdict as dd
from multiprocessing import Process, Queue

class Writer(Process):

  def __init__(self, cfg: Dict):
    super().__init__(daemon=True)

    self.queue = Queue()
    self.metrics = dd(dict)

    time = datetime.datetime.now().strftime("%c")
    self.path = cfg.config.log.save_path or f"log/{time}"
    self.load_path = cfg.config.log.load_path or self.path

    os.makedirs(self.path, exist_ok=True)
    shutil.copytree("nsm", f"{self.path}/src")

    with open(f"{self.path}/config.json", "w") as file:
      cfg_dict = dict(config=cfg.config, model=cfg.model, flow=cfg.flow)
      print(ConfigDict(cfg_dict).to_json(indent="\t"), file=file)

    if cfg.config.log.use_tensorboard:
      from tensorflow import summary
      self.tf_writer = summary.create_file_writer(self.path)
      self.tf_writer.set_as_default()

    self.files = {}

  def _file(self, name: str, keys: List[str]) -> TextIO:
    if name not in self.files:
      fname = f"{self.path}/{name}.csv"
  
      if os.path.isfile(fname):
        f = open(fname, "a")
      else:
        dir = os.path.dirname(fname)
        os.makedirs(dir, exist_ok=True)
        f = open(fname, "w")
        if keys is not None:
          csv.writer(f).writerow(keys)

      self.files[name] = f
    return self.files[name]

# ----------------------------------- IMPL ----------------------------------- #

  def load_checkpoint(self, it: Optional[int] = None) -> Tuple[int, Any]:
    """
      Load the model checkpoint at iteration `it`, or the latest
      checkpoint if `it` is None, or None if no checkpoint is found.
    """
    if it is None:
      for file in os.listdir(self.load_path):
        match = re.match(r"checkpoint.(\d+).pickle", file)
        if match: it = max(it or -1, int(match.group(1)))
    
    with open(fname:=f"{self.load_path}/checkpoint.{it}.pickle", "rb") as file:
      logging.info(f"Loading checkpoint {fname} ...")
      return it, pickle.load(file)

  def save_checkpoint(self, it: int, state: Any):
    """
      Save the checkpoint at iteration `it`.
    """
    with open(fname:=f"{self.path}/checkpoint.{it}.pickle", "wb") as file:
      logging.info(f"Saving checkpoint {fname} ...")
      pickle.dump(state, file)

  def write(self, it: int, metrics: Any):
    self.queue.put_nowait((it + 1, metrics))

  def finish(self):
    self.queue.put(None)

  # ---------------------------------------------------------------------------- #
  #                                  SUBPROCESS                                  #
  # ---------------------------------------------------------------------------- #

  def run(self):

    done = False
    while not done:

      for _ in range(max(1, self.queue.qsize())):
        if (item:=self.queue.get()) is None:
          done = True
          break
        else:
          it, metrics = item
          assert isinstance(metrics, dict)

          for name, metric in metrics.items():
            assert isinstance(metric, dict)

            for key, value in metric.items():
              log = self.metrics[name].setdefault(key, dd(list))
              log["it"].append(it)
              log["value"].append(value)

              if self.tf_writer is not None:
                from tensorflow import summary
                summary.scalar(f"{name}/{key}", value, it)

            file = self._file(name, ["it", *metric.keys()])
            csv.writer(file).writerow([it, *metric.values()])
            file.flush()

      for name, metric in self.metrics.items():
        fig, ax = plt.subplots(figsize=(4, 4))

        for key, log in metric.items():
          ax.loglog(log["it"], log["value"], label=key)

        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))
        fig.savefig(f"{self.path}/{name}.jpg", dpi=300)
        plt.close(fig)

    for file in self.files.values():
      file.close()

    self.tf_writer.close()
    logging.info(f"writer exit")
