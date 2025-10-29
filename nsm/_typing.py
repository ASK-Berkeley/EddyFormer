from typing import *
from typing_extensions import Self

import os, gc, sys
import time, datetime

import math
import numpy as np

import functools as F
import itertools as I

from absl import app, flags, logging
from dataclasses import dataclass, field

from collections import defaultdict as dd
from ml_collections import config_dict, ConfigDict
from ml_collections.config_dict import placeholder

from . import config

# ---------------------------------------------------------------------------- #
#                                      JAX                                     #
# ---------------------------------------------------------------------------- #

import jax

import flax
import optax

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.scipy.fft as jfft

from flax import struct
from flax import linen as nn
from flax import serialization

from jax.sharding import Mesh, PartitionSpec as P

# ------------------------------- EXPERIMENTAL ------------------------------- #

if jax.__version_info__ > (0, 6, 0):
  from jax import shard_map # 0.6.1
  shard_map = F.partial(shard_map, check_vma=False)
else:
  from jax.experimental.shard_map import shard_map
  shard_map = F.partial(shard_map, check_rep=False)

# ---------------------------------------------------------------------------- #
#                                    STRUCT                                    #
# ---------------------------------------------------------------------------- #

from jax import Array
Maybe = Optional

Shape = Tuple[int, ...]
PyTree = Union[Array, Sequence["PyTree"], Dict[str, "PyTree"]]
