import tensorflow as tf
import numpy as np
import argparse
import modeling_gpt2
import tensorflow_addons as tfa


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_ctx', default=512, type=int, required=False, help='')
    parser.add_argument('--model_config', default='configs/gpt2/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')
    parser.add_argument('--tfrecord_path', default='data/tokenized/tokenized.tfrecord', type=str, required=False,
                        help='预处理完成的数据地址')
    parser.add_argument('--lr', default=2e-4, type=float, required=False, help='学习率')
    parser.add_argument('--epochs', default=2, type=int, required=False, help='训练几个epoch')
    parser.add_argument('--steps_per_epoch', default=100, type=int, required=False, help='每个epoch多少步')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--save_interval', default=10, type=int, help='多少步保存一次模型')
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    class AutoSaveCallback(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            if (batch + 1) % args.save_interval == 0:
                self.model.save_pretrained(args.output_dir)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.writer_dir),
        AutoSaveCallback()
    ]
    print('getting dataset')
    feature_description = {
        'ids': tf.io.FixedLenFeature([args.n_ctx], tf.int64)}

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    ds = tf.data.TFRecordDataset(args.tfrecord_path)

    train_dataset = ds.map(_parse_function)

    def parse_2(example):
        return example['ids'][:-1], example['ids'][1:]

    train_dataset = train_dataset.map(parse_2)
    print('getting dataset done')
    # get dataset done
    print('total steps = {}'.format(args.epochs * args.steps_per_epoch))

    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True).shuffle(128).repeat(args.epochs)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print('starting training')
    with strategy.scope():
        model_config = modeling_gpt2.GPT2Config.from_json_file(args.model_config)
        if not args.pretrained_model:
            model = modeling_gpt2.TFGPT2LMHeadModel(config=model_config)
        else:
            model = modeling_gpt2.TFGPT2LMHeadModel.from_pretrained(args.pretrained_model)
        dummy = tf.constant(np.ones((args.batch_size, args.n_ctx)), dtype=tf.int32)
        _ = model([dummy])

        model.summary()
        optimizer = tfa.optimizers.RectifiedAdam(
            lr=args.lr,
            total_steps=args.epochs * args.steps_per_epoch,
            warmup_proportion=0.1,
            min_lr=1e-6,
        )
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer, loss_function)
    model.fit(train_dataset, callbacks=callbacks, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
    model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
