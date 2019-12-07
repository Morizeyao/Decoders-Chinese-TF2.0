import transformers
import torch
import tensorflow as tf
import os
import json
import random
import numpy as np
import argparse
import multiprocessing
import sentencepiece as spm
from multiprocessing import Process
from datetime import datetime
from tqdm import tqdm


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def build_tfrecord(raw_data_path, save_tfrecord_path, spm_model, min_length, n_ctx, batch_size, pad=0, epochs=1):
    def ids_example(ids):
        feature = {
            'ids': _int64_feature(ids),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    with open(raw_data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = f.readlines()
        lines = ['丨' + json.loads(line.rstrip('\n'))[0] + '丨' for line in tqdm(lines)]
        lines = [line for line in lines if len(line) > min_length]
    if not os.path.exists(save_tfrecord_path):
        os.makedirs(save_tfrecord_path)
    with tf.io.TFRecordWriter(save_tfrecord_path + 'tokenized.tfrecord') as f:
        for _ in range(epochs):
            random.shuffle(lines)
            for i in tqdm(range(len(lines) // batch_size)):
                batch = lines[i * batch_size:(i + 1) * batch_size]
                max_length_for_this_batch = 0
                for j, item in enumerate(batch):
                    batch[j] = spm_model.encode_as_ids(item)
                    max_length_for_this_batch = len(batch[j]) if len(batch[j]) > max_length_for_this_batch else max_length_for_this_batch
                for j, item in enumerate(batch):
                    while len(item) < max_length_for_this_batch:
                        item.append(pad)
                    batch[j] = item
                start_point = 0
                while start_point < max_length_for_this_batch - n_ctx:
                    for j in range(batch_size):
                        sample = batch[j][start_point:start_point + n_ctx]
                        assert len(sample) == n_ctx
                        # print(sample)
                        example = ids_example(sample)
                        f.write(example.SerializeToString())
                    start_point += n_ctx
                if start_point < max_length_for_this_batch:
                    for j in range(batch_size):
                        sample = batch[j][-n_ctx:]
                        while len(sample) < n_ctx:
                            sample.append(pad)
                        assert len(sample) == n_ctx
                        example = ids_example(sample)
                        f.write(example.SerializeToString())
    print('finish')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--spm_model_path', default='spm_model/ch.model', type=str, required=False, help='sentencepiece模型地址')
    parser.add_argument('--raw_data_path', default='data/train_test.txt', type=str, required=False, help='原始语料地址')
    parser.add_argument('--save_tfrecord_path', default='data/tokenized/', type=str, required=False, help='处理后的语料存放地址')
    parser.add_argument('--min_length', default=10, type=int, required=False, help='最短收录句子长度')
    parser.add_argument('--n_ctx', default=512, type=int, required=False, help='每个训练样本的长度')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='只用于XL模型，XL模型的batch size，GPT2设置为1')
    parser.add_argument('--pad', default=0, type=int, required=False, help='PAD值')
    parser.add_argument('--epochs', default=1, type=int, required=False, help='只用于XL模型，GPT2设置为1')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    ch_sp = spm.SentencePieceProcessor()
    ch_sp.Load(args.spm_model_path)

    build_tfrecord(args.raw_data_path, args.save_tfrecord_path, ch_sp, args.min_length, args.n_ctx,
                   args.batch_size, pad=args.pad, epochs=args.epochs)


if __name__ == '__main__':
    main()
