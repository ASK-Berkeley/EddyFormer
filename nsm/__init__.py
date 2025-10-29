from absl import flags

flags.DEFINE_bool("debug", False, "debug mode")
flags.DEFINE_bool("tqdm", True, "use tqdm")

FLAGS = flags.FLAGS

class config:

  @classmethod
  @property
  def debug(clf) -> bool:
    return FLAGS.debug

  @classmethod
  @property
  def tqdm(clf) -> bool:
    return FLAGS.tqdm
