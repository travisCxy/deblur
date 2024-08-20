import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
print(os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import re
from projects.uni import common

import random
import img_utils
import input_utils

MAX_WIDTH = 768
MAX_HEIGHT = 40

BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 512
def check_valid(text):
    cnt = 0
    for c in text:
        if c == '`':
            cnt += 1
    if cnt % 2 != 0:
        return False
    return True


def filter_fn(*args):
    _, label_tensor, cls_tensor, _ = args

    def should_keep(l, cls):
        l = l.decode('utf-8')
        # if '`' not in l:
        #     return False
        if cls == 2:
            l = '$' + l + '$'
        try:
            latexs = ["\\frac", "\\cdot", "\\times", "\\div", "\\sqrt{", "\\sqrt[", "\\leqslant", "\\triangle",
                      "\\left", "\\right", "\\because", "\\therefore", "\\angle", "^{", "_{", "\\mathrm", "\\pm",
                      "`o.`", "\\max", "\\cos", "\\sin"]
            for latex in latexs:
                if latex in l:
                    return False

            if u'²' in l or u'³' in l:
                return False

            v1 = common.code2attvec(l)
            #                return v1[0] != -1
            return v1[1] != common.END_SYMBOL
        except:
            # traceback.print_exc()
            print(l)
            return False

    # return tf.compat.v1.py_func(should_keep, [label_tensor, cls_tensor], tf.bool)
    return tf.compat.v1.py_func(should_keep, [label_tensor, cls_tensor], tf.bool)


def process_train_data(img, label, cls):

    img, w, seq_len = img_utils.img_to_dynamic_size(img,
                                                    MAX_WIDTH, MAX_HEIGHT,
                                                    pad_to_max=True, use_random=True)
    img = img_utils.image_aug_wrap(img)
    label = label.decode('utf-8')
    if cls == 2:
        label = '$' + label + '$'
    v1 = common.code2attvec(label)
    return img, np.int32(w), np.int32(seq_len), np.array(v1).astype(np.int32)


def train_fn(*args):
    image, label, cls, _ = args

    image, w, _, encoded_label = tf.compat.v1.py_func(process_train_data, [image, label, cls], [
        tf.uint8, tf.int32, tf.int32, tf.int32])
    image = tf.reshape(image, [MAX_HEIGHT, MAX_WIDTH, 3])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    return {"images": image, "labels": encoded_label}


def process_val_data(img, label, cls):
    img, w, seq_len = img_utils.img_to_dynamic_size(img,
                                                    MAX_WIDTH, MAX_HEIGHT,
                                                    fix_scale=1.0, pad_to_max=True, use_random=False)

    label = label.decode('utf-8')
    if cls == 2:
        label = '$' + label + '$'
    v1 = common.code2attvec(label)
    return img, np.int32(w), np.int32(seq_len), np.array(v1).astype(np.int32)


def val_fn(*args):
    image, label, cls, _ = args

    image, w, _, encoded_label = tf.compat.v1.py_func(process_val_data, [image, label, cls], [
        tf.uint8, tf.int32, tf.int32, tf.int32])
    image = tf.reshape(image, [MAX_HEIGHT, MAX_WIDTH, 3])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    return {"images": image, "labels": encoded_label, "org_labels": label}


def create_dataset(data_partten, training):
    dataset = input_utils.get_inputs(data_partten, training, train_fn=train_fn, val_fn=val_fn, filter_fn=filter_fn)
    if training:
        dataset = dataset.padded_batch(batch_size=TRAIN_BATCH_SIZE,
                                       padded_shapes={
                                            "images": [MAX_HEIGHT, None, 3],
                                            "labels": [common.MAX_SEQ_LEN]
                                       },
                                       drop_remainder=True)
        return dataset.prefetch(4)
    else:
        dataset = dataset.batch(batch_size=EVAL_BATCH_SIZE, drop_remainder=False)
        return dataset.prefetch(1)

