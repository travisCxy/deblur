# -*-coding:utf-8-*-
import tensorflow as tf
import cv2

import glob
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

def read_tfrecord(data_path_partten):
    dataset = tf.data.Dataset.list_files(data_path_partten, shuffle=False)
    dataset = dataset.repeat(1)
    dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename),
                                 cycle_length=1)
    dataset = dataset.map(parse_example)
    return dataset

files = glob.glob('/mnt/server_data/data/seqhw/tfrecords/train_hw_rotated_digits*')
data = read_tfrecord(files)
for i, (image, label, cls, filename) in enumerate(data):
    print(i)
    print(image.shape)
    print(label)
    print(cls)
    print(filename)
    print('=====================')
    cv2.imwrite(f'./{i}.jpg', image.numpy())
    if i == 10:
        break