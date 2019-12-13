python3 train_gpt2_keras.py \
  --n_ctx 512 \
  --model_config configs/gpt2/model_config_small.json \
  --pretrained_model '' \
  --batch_size 1 \
  --tfrecord_path data/tokenized/tokenized.tfrecord \
  --lr 2e-4 \
  --epochs 10 \
  --steps_per_epoch 100 \
  --output_dir model/
