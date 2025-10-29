from nsm.typing import *
from configs.flow import default

def get_config():
  cfg = default()
  cfg.file = __file__

  cfg.path = "homogeneous.Isotropic"
  cfg.config = ConfigDict({
    "base_dir": placeholder(str),
    "name": "re94",
    "nu": 0.01,
    "t": 0.5,
    # Forcing config
    "mode": 1,
    "P_in": 1.0,
    # IC config
    "initial_condition": ConfigDict({
      "t0": 20.0,
      "scale": 3.0,
      "resolution": (96, 96, 96),
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
    "solver_res": (96, 96, 96),
    "correction": 1e-5,
  })

  return cfg
