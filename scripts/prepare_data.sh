python3 prepare_data.py \
    --spm_model_path spm_model/ch.model \
    --raw_data_path data/train_test.txt \
    --save_tfrecord_path data/tokenized/ \
    --min_length 10 \
    --n_ctx 512 \
    --batch_size 1 \
    --pad 0 \
    --epochs 1