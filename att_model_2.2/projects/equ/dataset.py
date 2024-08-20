import tensorflow as tf
import numpy as np

import input_utils
import img_utils

from projects.equ import common

MAX_WIDTH = 256
MAX_HEIGHT = 40

TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 512


def filter_fn(*args):
    _, label_tensor, cls_tensor, _ = args

    def should_keep(l, cls):
        l = l.decode('utf-8')
        if cls not in [1, 3, 4, 6, 7]:
            return False

        if "." in l:
            return True

        if "[" in l:
            return True
        return True

    return tf.compat.v1.py_func(should_keep, [label_tensor, cls_tensor], tf.bool)


def process_train_data(img, label):
    label = label.decode('utf-8')
    fix_size = (MAX_WIDTH, MAX_HEIGHT)
    img = img_utils.img_to_fix_size(img, fix_size, use_random=True, method="all")
    img = img_utils.image_aug_wrap(img, prob=0.2)
    l1, l2, l3 = common.parse_frac(label)
    weight = 1.0  # 0.0 if l2[0] == common.FRAC_SYMBOL or l3[0] == common.FRAC_SYMBOL else 1.0
    return (img,
            np.array([common.code2vec(l1), common.code2vec(l2), common.code2vec(l3)]).astype(np.int32),
            np.float32(weight))


def train_fn(*args):
    image, label, cls, filename = args
    image, label, weight = tf.compat.v1.py_func(process_train_data, [image, label], [
                                                       tf.uint8, tf.int32, tf.float32])

    image = tf.reshape(image, [MAX_HEIGHT, MAX_WIDTH, 3])
    label = tf.reshape(label, [3, common.MAX_SEQ_LEN])
    weight = tf.reshape(weight, [])
    seq_len = tf.constant(MAX_WIDTH // 4, dtype=tf.int32)
    return {"images": image, "labels": label, "weights": weight, "seq_lens": seq_len}


def process_val_data(img):
    fix_size = (MAX_WIDTH, MAX_HEIGHT)
    scale_img = img_utils.img_to_fix_size(img, fix_size, use_random=False, method="scale")
    pad_img = img_utils.img_to_fix_size(img, fix_size, use_random=False, method="pad")
    return scale_img, pad_img


def val_fn(*args):
    image, label, cls, filename = args

    scale_img, pad_img = tf.compat.v1.py_func(process_val_data, [image], [tf.uint8, tf.uint8])
    scale_img = tf.reshape(scale_img, [MAX_HEIGHT, MAX_WIDTH, 3])
    pad_img = tf.reshape(pad_img, [MAX_HEIGHT, MAX_WIDTH, 3])
    seq_len = tf.constant(MAX_WIDTH // 4, dtype=tf.int32)
    images = tf.stack([scale_img, pad_img], axis=0)
    seq_lens = tf.stack([seq_len, seq_len], axis=0)
    labels = tf.stack([label, label], axis=0)
    filename = tf.stack([filename, filename], axis=0)
    return {"images": images, "labels": labels, "seq_lens": seq_lens, "filenames": filename}


def create_dataset(data_partten, training):
    dataset = input_utils.get_inputs(data_partten, training, train_fn=train_fn, val_fn=val_fn, filter_fn=filter_fn)
    if training:
        dataset = dataset.batch(batch_size=TRAIN_BATCH_SIZE, drop_remainder=True)
        return dataset.prefetch(4)
    else:
        dataset = dataset.batch(batch_size=EVAL_BATCH_SIZE, drop_remainder=False)
        return dataset.prefetch(1)
