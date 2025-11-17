# EddyFormer

EddyFormer is a Transformer-based spectral element model for turbulence simulation.

![demo](re94-demo.gif)

EddyFormer uses SEM tokenization to represent grid- and subgrid-scale dynamics, combining spectral accuracy with scalable attention mechanism. Trained on a 3D isotropic turbulence dataset, EddyFormer matches DNS accuracy at 256³ while running 30× faster. For more details, see our paper: https://arxiv.org/abs/2510.24173.

## Dependencies

Install the package and its dependencies:
```
pip install -e .
```

You might need to manually install the latest version of JAX:
```
pip install --upgrade "jax[cuda12]"
```
**Note**: EddyFormer has been tested with JAX 0.6.0.
If you are using a JAX version older than 0.4.x, multi-GPU training may not be supported.
You can disable it by adding a config:
```
--config.train.batch_sharding=False
```

## Dataset

The Re94 dataset is publicly available on [Hugging Face](https://huggingface.co/datasets/ydu11/re94).
You can download it with:
```
hf download --repo-type dataset ydu11/re94 --local-dir data/ns3d-re94
```

Once downloaded, initialize the flow field as follows:
```python
from nsm.flow import Isotropic
from configs.flow import re94

config = re94.get_config().config
flow = Isotropic(**config)
```

## Model

To instantiate the EddyFormer model:
```python
from nsm.model import EddyFormer
from configs.model import ef3d

config = ef3d.get_config().config
model = EddyFormer(**config)
```

You can initialize the model weights using a dataset sample:
```python
from jax.random import PRNGKey
u = next(iter(flow.dataset("test")))
variable = model.init(PRNGKey(42), flow, u[0].ic)
```

## Experiments

All training scripts are provided in the scripts directory.
For example, to train EddyFormer on Re94 using a Legendre basis, run:
```
sbatch scripts/re94.sbatch leg
```

## Citation

If you use EddyFormer in your research, please cite the corresponding paper:
```
@inproceedings{
  du2025eddyformer,
  title={EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale},
  author={Yiheng Du and Aditi S. Krishnapriyan},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2510.24173}
}
```
