#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 24:00:00
#SBATCH -o log/slurm/%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpu # <your conda env name>

export logdir=log/kf4/ef-"$1"-t5
export XLA_FLAGS="--xla_force_host_platform_device_count=16"

rm -r $logdir/* || mkdir $logdir
ln -s $(pwd)/log/slurm/$SLURM_JOBID.out $logdir/slurm.out

TERM="unknown" srun -u \
nsm train --notqdm \
  --flow configs/flow/kf4.py \
  --flow.config.solver.max_steps 400 \
  --model configs/model/ef2d.py \
  --model.config.basis "$1"_elem \
  --config.train.vmap_batch 4 \
  --config.train.window 5 \
  --config.train.iteration 45000 \
  --config.train.learning_rate 1e-5 \
  --config.log.load_path log/kf4/ef-"$1" \
  --config.log.save_path $logdir \
| tee $logdir/stdout.log 2>&1 \
| tee $logdir/stderr.log
