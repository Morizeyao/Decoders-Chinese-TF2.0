import transformers
import tensorflow as tf
import os
import numpy as np
import argparse
from datetime import datetime


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_ctx', default=512, type=int, required=False, help='文本长度')
    parser.add_argument('--model_config', default='configs/xl/config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')
    parser.add_argument('--tfrecord_path', default='data/tokenized/tokenized.tfrecord', type=str, required=False,
                        help='预处理完成的数据地址')
    parser.add_argument('--lr', default=2e-4, type=float, required=False, help='学习率')
    parser.add_argument('--total_steps', default=10, type=int, required=False, help='steps')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步报告一次')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    summary_writer = tf.summary.create_file_writer(args.writer_dir)
    print('getting dataset')
    feature_description = {
        'ids': tf.io.FixedLenFeature([args.n_ctx], tf.int64)}

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    ds = tf.data.TFRecordDataset(args.tfrecord_path)

    train_dataset = ds.map(_parse_function)
    print('getting dataset done')
    # get dataset done
    print('total steps = {}'.format(args.total_steps))

    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print('starting training')
    with strategy.scope():
        model_config = transformers.modeling_tf_transfo_xl.TransfoXLConfig.from_json_file(args.model_config)
        if not args.pretrained_model:
            model = transformers.modeling_tf_transfo_xl.TFTransfoXLLMHeadModel(config=model_config)
        else:
            model = transformers.modeling_tf_transfo_xl.TFTransfoXLLMHeadModel.from_pretrained(args.pretrained_model)
        dummy = tf.constant(np.ones((args.batch_size, args.n_ctx)), dtype=tf.int32)
        # print(dummy.shape)
        _ = model([dummy])

        model.summary()
        # print(model.trainable_variables)

        scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.lr,
                                                                  decay_steps=args.total_steps,
                                                                  end_learning_rate=args.lr * 0.01)
        optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_function(labels, predictions)
            # tf.print(per_example_loss)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)


        running_loss = 0
        step = 0
        epoch = 0


        print('saving initial model')
        if not os.path.exists(args.output_dir + 'model_temp_step{}'.format(step)):
            os.makedirs(args.output_dir + 'model_temp_step{}'.format(step))
        model.save_pretrained(args.output_dir + 'model_temp_step{}'.format(step))


        mems = None
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        while True:
            for batch_idx, batch_inputs in enumerate(iter(train_dataset)):
                batch_inputs = batch_inputs['ids']


                def train_step(batch_inputs, mems):
                    inputs, labels = batch_inputs, batch_inputs[:, 1:]

                    with tf.GradientTape() as tape:
                        outputs, mems = model([batch_inputs, mems], training=True)[:2]
                        loss = compute_loss(labels, outputs[:, :-1, :])
                    gradients = tape.gradient(loss, model.trainable_variables)
                    print('gradients of first trainable var:')
                    tf.print(gradients[0])
                    tvars = list({id(v): v for v in model.trainable_variables}.values())
                    optimizer.apply_gradients(zip(gradients, tvars))
                    return loss, mems

                @tf.function
                def get_loss(work, batch_inputs, mems):
                    per_replica_losses, mems = strategy.experimental_run_v2(work,
                                                                            args=(batch_inputs, mems))
                    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                           axis=None)
                    return loss, mems

                loss, mems = get_loss(train_step, batch_inputs, mems)
                running_loss += loss
                if (step + 1) % args.log_step == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar("loss", running_loss / args.log_step, step=step)
                        summary_writer.flush()
                    print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        batch_idx + 1,
                        epoch + 1,
                        running_loss / args.log_step))
                    running_loss = 0
                step += 1
                if step > args.total_steps:
                    break
                if (step + 1) % 10000 == 0:
                    print('saving model temp')
                    if not os.path.exists(args.output_dir + 'model_temp_step{}'.format(step)):
                        os.makedirs(args.output_dir + 'model_temp_step{}'.format(step))
                    model.save_pretrained(args.output_dir + 'model_temp_step{}'.format(step))
            epoch += 1

            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(args.output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.makedirs(args.output_dir + 'model_epoch{}'.format(epoch + 1))
            model.save_pretrained(args.output_dir + 'model_epoch{}'.format(epoch + 1))
            print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

        print('training finished')
        if not os.path.exists(args.output_dir + 'final_model'):
            os.makedirs(args.output_dir + 'final_model')
        model.save_pretrained(args.output_dir + 'final_model')


if __name__ == '__main__':
    main()
