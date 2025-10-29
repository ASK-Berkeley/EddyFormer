#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 24:00:00
#SBATCH -o log/slurm/%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpu # <your conda env name>

export logdir=log/re94/ef-"$1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_FLAGS="--xla_force_host_platform_device_count=16"

rm -r $logdir/* || mkdir $logdir
ln -s $(pwd)/log/nersc-nsm/$SLURM_JOBID.out $logdir/slurm.out

TERM="unknown" srun -u \
nsm train --notqdm \
  --flow configs/flow/re94.py \
  --model configs/model/ef3d.py \
  --model.config.basis "$1"_elem \
  --config.train.vmap_batch 1 \
  --config.train.iteration 15000 \
  --config.log.save_path $logdir \
>  >(tee $logdir/stdout.log) \
2> >(tee $logdir/stderr.log >&2)
