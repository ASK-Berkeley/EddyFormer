#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 24:00:00
#SBATCH -o log/slurm/%j.out

rm -r $logdir/* || mkdir $logdir
ln -s $(pwd)/log/slurm/$SLURM_JOBID.out $logdir/slurm.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpu # <your conda env name>

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_FLAGS="--xla_force_host_platform_device_count=16"

TERM="unknown" srun -u \
nsm train --notqdm \
  --flow configs/flow/the_well.py \
  --config.train.iteration 10000 \
  --config.log.test_period 10000 \
  --config.log.save_path $logdir \
  "$@" \
| tee $logdir/stdout.log 2>&1 \
| tee $logdir/stderr.log
