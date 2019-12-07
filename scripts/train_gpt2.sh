python3 train_gpt2.py \
  --n_ctx 512 \
  --model_config configs/gpt2/model_config_small.json \
  --pretrained_model '' \
  --batch_size 2 \
  --tfrecord_path data/tokenized/tokenized.tfrecord \
  --lr 2e-4 \
  --total_steps 100 \
  --output_dir model/