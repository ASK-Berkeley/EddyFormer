export name=shear_flow

logdir=log/the_well/"$name"/ef-"$1" \
sbatch scripts/the_well/_base.sbatch \
  --flow.config.name "$name" \
  --model configs/model/ef2d.py \
  --model.config.odim 4 \
  --model.config.basis "$1"_elem \
  --model.config.mesh "(16, 32)" \
  --config.train.batch_size 64
