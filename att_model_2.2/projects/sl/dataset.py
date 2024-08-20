import tensorflow as tf

import numpy as np

from projects.sl import common

import random
import img_utils
import input_utils

MAX_WIDTH = 512
MAX_HEIGHT = 48

BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512


def filter_fn(*args):
    _, label_tensor, _, _ = args

    def should_keep(l):
        try:
            l = l.decode('utf-8')
            # shushi divide mark error
            if "`" in l and 'D' in l:
                return False
            common.code2attvec(l)
            return True
        except:
            import traceback
            traceback.print_exc()
            print(l)
            return False

    return tf.compat.v1.py_func(should_keep, [label_tensor], tf.bool)


def process_train_data(img, label):
    def random_crop(img):
        if random.randint(0, 9) <= 4:
            return img

        h, w = img.shape[:2]

        fixed_crop_size = 4
        scale_rate = h * 1.0 / MAX_HEIGHT
        crop_size = int(fixed_crop_size * scale_rate)
        for _ in range(5):
            crop_type = random.randint(0, 2)
            left, right, top, bottom = 0, 0, 0, 0
            if crop_type in [0, 2]:
                left = random.randint(0, crop_size)
                right = random.randint(0, crop_size - left)
            if crop_type in [1, 2]:
                top = random.randint(0, crop_size)
                bottom = random.randint(0, crop_size - top)

            if top + 20 < h - bottom and left + 20 < w - right:
                break
            else:
                top, bottom, left, right = 0, 0, 0, 0

        img = img[top:h - bottom + 1, left:w - right + 1, :]

        return img

    # img = random_crop(img)
    img, _, seq_len = img_utils.img_to_dynamic_size(img,
                                                    MAX_WIDTH, MAX_HEIGHT,
                                                    pad_to_max=True, use_random=True)
    img = img_utils.image_aug_wrap(img)

    label = label.decode('utf-8')
    v1 = common.code2attvec(label)
    return img, seq_len, np.array(v1).astype(np.int32)


def train_fn(*args):
    image, label, cls, _ = args

    image, _, encoded_label = tf.compat.v1.py_func(process_train_data, [image, label], [
        tf.uint8, tf.int32, tf.int32])
    image = tf.reshape(image, [MAX_HEIGHT, MAX_WIDTH, 3])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    return {"images": image, "labels": encoded_label}


def process_val_data(img, label):
    img, w, seq_len = img_utils.img_to_dynamic_size(img,
                                                    MAX_WIDTH, MAX_HEIGHT,
                                                    fix_scale=1.0, pad_to_max=True, use_random=False)

    label = label.decode('utf-8')
    v1 = common.code2attvec(label)
    return img, w, seq_len, np.array(v1).astype(np.int32)


def val_fn(*args):
    image, label, cls, _ = args

    image, w, _, encoded_label = tf.compat.v1.py_func(process_val_data, [image, label], [
        tf.uint8, tf.int64, tf.int32, tf.int32])
    image = tf.reshape(image, [MAX_HEIGHT, MAX_WIDTH, 3])
    encoded_label = tf.reshape(encoded_label, [common.MAX_SEQ_LEN])
    return {"images": image, "norm_ws": w, "labels": encoded_label, "org_labels": label}


def create_dataset(data_partten, training):
    dataset = input_utils.get_inputs(data_partten, training, train_fn=train_fn, val_fn=val_fn, filter_fn=filter_fn)
    if training:
        dataset = dataset.batch(batch_size=TRAIN_BATCH_SIZE, drop_remainder=True)
        return dataset.prefetch(4)
    else:
        dataset = dataset.batch(batch_size=EVAL_BATCH_SIZE, drop_remainder=False)
        return dataset.prefetch(1)
