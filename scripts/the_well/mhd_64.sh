export name=MHD_64

logdir=log/the_well/"$name"/ef-"$1" \
sbatch scripts/the_well/_base.sbatch \
  --flow.config.name "$name" \
  --model configs/model/ef3d.py \
  --model.config.odim 7 \
  --model.config.basis "$1"_elem \
  --config.train.batch_size 64 \
  --config.train.gradient_clip 1.0
