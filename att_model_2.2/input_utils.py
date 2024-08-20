import tensorflow as tf

import glob
import sys

import horovod.tensorflow as hvd


def image_process_for_train(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = (image - 0.5) * 2
    return image


def image_process_for_val(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 0.5) * 2
    return image


def parse_example(example):
    features = tf.io.parse_single_example(example,
                                          features={
                                              'image/name': tf.io.FixedLenFeature([], tf.string),
                                              'image/label': tf.io.FixedLenFeature([], tf.string),
                                              'image/encoded': tf.io.FixedLenFeature([], tf.string),
                                              'image/cls': tf.io.FixedLenFeature([], tf.int64)
                                          })
    cls = features['image/cls']
    label = features['image/label']
    image = tf.image.decode_jpeg(features['image/encoded'])
    filename = features['image/name']

    return image, label, cls, filename


def get_inputs(data_path_partten, training, train_fn, val_fn, filter_fn, parse_fn=parse_example):
    dataset = tf.data.Dataset.list_files(data_path_partten, shuffle=False)
    # for num, _ in enumerate(dataset):
    #     pass
    # print(f'Number of elements: {num}')
    count = tf.data.experimental.cardinality(dataset).numpy()
    print(dataset)
    print("================== total count: %d ==============" % count)
    if training:
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.shuffle(count // hvd.size(), reshuffle_each_iteration=True)
        dataset = dataset.repeat(-1)
        dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename),
                                     cycle_length=8)
        dataset = dataset.shuffle(10000)
    else:
        dataset = dataset.repeat(1)
        dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename),
                                     cycle_length=1)

    dataset = dataset.map(parse_fn)
    dataset = dataset.filter(filter_fn)
    transfer_fn = train_fn if training else val_fn
    # dataset = dataset.map(transfer_fn, num_parallel_calls=4)
    dataset = dataset.map(transfer_fn, num_parallel_calls=4)
    return dataset
