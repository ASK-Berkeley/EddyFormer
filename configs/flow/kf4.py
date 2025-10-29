from nsm.typing import *
from configs.flow import default

"""
Statistics:
  - Energy: 2.0061815
  - Dispassion: 0.041994397
  - Velocity R.M.S.: 1.1538857
  - Taylor micro. scale: 0.6938955
  - Renolds No. at T.M.S.: 799.7589
"""

def get_config():
  cfg = default()

  cfg.path = "homogeneous.KolmogorovFlow"
  cfg.config = ConfigDict({
    "base_dir": placeholder(str),
    "nu": 0.001,
    "t": 1.0,
    "s": 1,
    # KF config
    "mode": 4,
    "alpha": 0.1,
    # IC config
    "initial_condition": ConfigDict({
      "t0": 40.0,
      "scale": 7.0,
      "resolution": (256, 256),
      "wavenumber": 16,
    }),
    # Solver config
    "solver": ConfigDict({
      "courant": 1.0,
      "dt": placeholder(float),
      "dt_scheme": "rk4",
      "max_steps": placeholder(int),
      "dealias": "smooth",
    }),
    # Learned correction
    "solver_res": (256, 256),
    "correction": 1e-5,
  })

  return cfg
