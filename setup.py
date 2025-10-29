from setuptools import setup

setup(
  name="nsm",
  version="0.1",
  packages=["nsm"],
  scripts=[
    "bin/nsm",
  ],
  python_requires=">=3.10",
  install_requires=[
   # JAX
    "jax[cuda12]",
    "flax",
    "optax",
    "quadax",
   # PACKAGE
    "absl-py",
    "ml_collections",
   # LOGGING
    "tqdm",
    "tensorflow-cpu",
    "matplotlib",
    "scienceplots",
   # DEBUGGING
    "vtk",
    "nvtx",
  ],
)
