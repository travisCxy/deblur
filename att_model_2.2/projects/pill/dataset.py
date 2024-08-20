import tensorflow as tf
import numpy as np

import input_utils
import img_utils

from projects.pill import common

MAX_WIDTH = 160
MAX_HEIGHT = 40

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256


def filter_fn(*args):
    _, label_tensor, _, _ = args

    def should_keep(l):
        l = l.decode('utf-8')
        try:
            v1 = common.code2vec(l)
            return True
        except:
            import traceback
            traceback.print_exc()
            print(l)
            return False

    return tf.compat.v1.py_func(should_keep, [label_tensor], tf.bool)


def process_train_data(img, label):
    label = label.decode('utf-8')
    img, w, seq_len = img_utils.img_to_dynamic_size(img, MAX_WIDTH, MAX_HEIGHT,
                                                      pad_to_max=False, use_random=True)
    #img = img_utils.image_aug_wrap(img)
    v1 = common.code2vec(label)
    return img, seq_len, np.int32(w), np.array(v1).astype(np.int32)


def train_fn(*args):
    image, label, cls, filename = args
    image, seq_len, w, encoded_label = tf.compat.v1.py_func(process_train_data, [image, label], [
                                                       tf.uint8, tf.int32, tf.int32, tf.int32])

    image = tf.reshape(image, [MAX_HEIGHT, w, 3])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    seq_len = tf.reshape(seq_len, [])
    return {"images": image, "labels": encoded_label, "seq_lens": seq_len}


def process_val_data(img, label):
    label = label.decode('utf-8')
    img, w, seq_len = img_utils.img_to_dynamic_size(img, MAX_WIDTH, MAX_HEIGHT,
                                                      fix_scale=1.0, pad_to_max=False, use_random=False)
    v1 = common.code2vec(label)
    return img, seq_len, np.int32(w), np.array(v1).astype(np.int32)


def val_fn(*args):
    image, label, cls, filename = args

    image, seq_len, w, encoded_label = tf.compat.v1.py_func(process_val_data, [image, label], [
                                                       tf.uint8, tf.int32, tf.int32, tf.int32])
    image = tf.reshape(image, [MAX_HEIGHT, w, 3])
    seq_len = tf.reshape(seq_len, [])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    return {"images": image, "labels": encoded_label, "seq_lens": seq_len, "filenames": filename}


def create_dataset(data_partten, training):
    dataset = input_utils.get_inputs(data_partten, training, train_fn=train_fn, val_fn=val_fn, filter_fn=filter_fn)
    if training:
        dataset=dataset.padded_batch(batch_size=TRAIN_BATCH_SIZE,
                                     padded_shapes={"images": [MAX_HEIGHT, None, 3],
                                                    "labels": [common.MAX_SEQ_LEN],
                                                    "seq_lens": []},
                                     drop_remainder=True)
        return dataset.prefetch(4)
    else:
        dataset = dataset.padded_batch(batch_size=EVAL_BATCH_SIZE,
                                       padded_shapes={"images": [MAX_HEIGHT, None, 3],
                                                      "labels": [common.MAX_SEQ_LEN],
                                                      "seq_lens": [],
                                                      "filenames": []},
                                       drop_remainder=False)
        return dataset.prefetch(1)
