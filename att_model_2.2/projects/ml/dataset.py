import tensorflow as tf

import numpy as np

from projects.ml import common

import random
import img_utils
import input_utils

MAX_WIDTH = 384
MAX_HEIGHT = 120

BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 512


def parse_example(example):
    features = tf.io.parse_single_example(example,
                                          features={
                                              'image/name': tf.io.FixedLenFeature([], tf.string),
                                              'image/label': tf.io.FixedLenFeature([], tf.string),
                                              'image/encoded': tf.io.FixedLenFeature([], tf.string),
                                              'image/cls': tf.io.FixedLenFeature([], tf.int64),
                                              'image/rect': tf.io.FixedLenFeature([4], tf.int64),
                                          })
    cls = features['image/cls']
    label = features['image/label']
    image = tf.image.decode_jpeg(features['image/encoded'])
    filename = features['image/name']
    rect = features["image/rect"]
    return image, label, cls, rect, filename


def filter_fn(*args):
    _, label_tensor, cls_tensor, _, name_tensor = args

    def should_keep(l, name):
        l = l.decode('utf-8')
        try:
            if l == '':
                return False
            v1 = common.code2attvec(l)
            return v1[1] != common.END_SYMBOL
        except:
            import traceback
            import sys
            traceback.print_exc()
            print(l)
            sys.stdout.flush()
            return False

    return tf.compat.v1.py_func(should_keep, [label_tensor, name_tensor], tf.bool)


def random_crop_img(img, rect):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    max_left = x1
    max_top = y1
    max_right = w - x2
    max_bottom = h - y2

    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    right = random.randint(0, max_right)
    bottom = random.randint(0, max_bottom)

    x1 = x1 - left
    y1 = y1 - top
    x2 = x2 + right
    y2 = y2 + bottom

    ret = img[y1:y2, x1:x2]
    return ret


def process_train_data(img, label, rect):
    img = random_crop_img(img, rect)
    img = img_utils.img_to_fix_size(img, (MAX_WIDTH, MAX_HEIGHT), use_random=True, method="pad")
    #img = img_utils.image_aug_wrap(img)
    label = label.decode('utf-8')
    v1 = common.code2attvec(label)
    return img, np.array(v1).astype(np.int32)


def train_fn(*args):
    image, label, cls, rect, _ = args

    image, encoded_label = tf.compat.v1.py_func(process_train_data, [image, label,  rect], [
        tf.uint8,  tf.int32])
    image = tf.reshape(image, [MAX_HEIGHT, MAX_WIDTH, 3])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    return {"images": image, "labels": encoded_label}


def process_val_data(img, label):
    img = img_utils.img_to_fix_size(img, (MAX_WIDTH, MAX_HEIGHT), use_random=False, method="pad")
    label = label.decode('utf-8')
    v1 = common.code2attvec(label)
    return img, np.array(v1).astype(np.int32)


def val_fn(*args):
    image, label, cls, _, _ = args

    image, encoded_label = tf.compat.v1.py_func(process_val_data, [image, label], [
        tf.uint8, tf.int32])
    image = tf.reshape(image, [MAX_HEIGHT, MAX_WIDTH, 3])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    return {"images": image, "labels": encoded_label, "org_labels": label}


def create_dataset(data_partten, training):
    dataset = input_utils.get_inputs(data_partten, training, train_fn=train_fn, val_fn=val_fn,
                                     filter_fn=filter_fn, parse_fn=parse_example)
    if training:
        dataset = dataset.batch(batch_size=TRAIN_BATCH_SIZE, drop_remainder=True)
        return dataset.prefetch(4)
    else:
        dataset = dataset.batch(batch_size=EVAL_BATCH_SIZE, drop_remainder=False)
        return dataset.prefetch(1)

if __name__ == '__main__':
    import cv2
    from horovod import tensorflow as hvd
    hvd.init()
    dataset = create_dataset("/mnt/data1/heping/mathlens/tfrecords/train-mathlens.tfrecords-*", training=True)
    for items in dataset:
        #pass
        imgs = items["images"]
        for img in imgs.numpy():
            cv2.imshow("debug", img)
            cv2.waitKey()