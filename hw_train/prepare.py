from absl import flags, app
import tensorflow as tf
import glob
import os
import numpy as np
import cv2
from utils import tfrecord_writer
import common
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# flags.DEFINE_string('data_dir', "datasets", 'data dir.')
flags.DEFINE_string('data_dir', "datasets/train/20240524_by/raw_data", 'data dir.')
flags.DEFINE_string('target_dir', "datasets/train/20240524_by/tfrecords", 'target dir.')
flags.DEFINE_string('prefix', 'train-tianzige_0528', 'train/val.')
#
# flags.DEFINE_string('data_dir', "datasets/raw_data/20231020/train_A", 'data dir.')
# flags.DEFINE_string('target_dir', "datasets/train/20240513/tfrecords", 'target dir.')
# flags.DEFINE_string('prefix', 'train-rescale', 'train/val.')
ignored_cnt = 0


def read_file(source_path, target_path):
    return tf.io.read_file(source_path), tf.io.read_file(target_path)


def split_4(img):
    h, w, _ = img.shape
    target_h = int((h + 1) / 2) * 2
    target_w = int((w + 1) / 2) * 2
    img = tf.image.pad_to_bounding_box(img, 0, 0, target_h, target_w)
    s1, s2 = tf.split(img, 2, axis=0)
    s11, s12 = tf.split(s1, 2, axis=1)
    s21, s22 = tf.split(s2, 2, axis=1)
    return s11, s12, s21, s22


def split_and_save(tfrecord_gen, source_img, target_img):
    h, w, _ = source_img.shape
    global ignored_cnt
    if h >= 1024 and w >= 1024:
        source_arr = split_4(source_img)
        target_arr = split_4(target_img)
        for i in range(4):
            split_and_save(tfrecord_gen, source_arr[i], target_arr[i])
    else:
        if source_img.shape != target_img.shape:
            print(source_img.shape, target_img.shape)
            raise ValueError("AAAAAAAAA")
        if h<300 or w<300:
            ignored_cnt += 1
            print("ignored", ignored_cnt)
            return
        tfrecord_gen.write_image_and_image(tf.image.encode_jpeg(tf.cast(source_img, tf.uint8)).numpy(),
                                           tf.image.encode_jpeg(tf.cast(target_img, tf.uint8)).numpy())

# def split_and_save(tfrecord_gen, source_img, target_img):
#     h, w, _ = source_img.shape
#     global ignored_cnt
#     if h >= 1024 and w >= 1024:
#         source_arr = split_4(source_img)
#         target_arr = split_4(target_img)
#         for i in range(4):
#             split_and_save(tfrecord_gen, source_arr[i], target_arr[i])
#     else:
#         if source_img.shape != target_img.shape:
#             print(source_img.shape, target_img.shape)
#             raise ValueError("AAAAAAAAA")
#         if h<300 or w<300:
#             ignored_cnt += 1
#             print("ignored", ignored_cnt)
#             return
#         tfrecord_gen.write_image_and_image(tf.image.encode_jpeg(tf.cast(source_img, tf.uint8)).numpy(),
#                                            tf.image.encode_jpeg(tf.cast(target_img, tf.uint8)).numpy())


def split_and_save3(tfrecord_gen, source_img, target_img):
    source_img = cv2.imread(source_img)
    target_img = cv2.imread(target_img)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    h, w, _ = source_img.shape
    global ignored_cnt

    if source_img.shape != target_img.shape:
        print(source_img.shape, target_img.shape)
        raise ValueError("AAAAAAAAA")
    if h<300 or w<300:
        ignored_cnt += 1
        print("ignored", ignored_cnt)
        return
    h,w = source_img.shape[:2]
    new_h = h
    new_w = w
    if h%32!= 0:
        new_h = (h//32+1)*32
    if w%32!= 0:
        new_w = (w//32+1)*32
    if random.random() > 0.5:
        new_source_img = cv2.resize(source_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        new_target_img = cv2.resize(target_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        source_img = new_source_img
        target_img = new_target_img
    else:
        pad_source_img = np.ones((new_h, new_w, 3), dtype=np.uint8)*255
        pad_target_img = np.ones((new_h, new_w, 3), dtype=np.uint8)*255
        pad_source_img[:h, :w, :] = source_img
        pad_target_img[:h, :w, :] = target_img
        source_img = pad_source_img
        target_img = pad_target_img
    source_img = tf.convert_to_tensor(source_img)
    target_img = tf.convert_to_tensor(target_img)
    tfrecord_gen.write_image_and_image(tf.image.encode_jpeg(tf.cast(source_img, tf.uint8)).numpy(),
                                       tf.image.encode_jpeg(tf.cast(target_img, tf.uint8)).numpy())


def split_and_save_2(tfrecord_gen, source_img):
    h, w, _ = source_img.shape
    global ignored_cnt
    if h >= 1024 and w >= 1024:
        source_arr = split_4(source_img)
        for i in range(4):
            split_and_save_2(tfrecord_gen, source_arr[i])
    else:
        tfrecord_gen.write_image_and_label(tf.image.encode_jpeg(tf.cast(source_img, tf.uint8)).numpy(),
                                           tf.image.encode_jpeg(tf.cast(source_img, tf.uint8)).numpy(),
                                           1)


def main(_):
    if flags.FLAGS.prefix == "train" or "train" in flags.FLAGS.prefix:
        # source_files = glob.glob(flags.FLAGS.data_dir + "/train/20231113_水彩/raw_data3/*_crop.png", recursive=True)
        source_files = glob.glob(flags.FLAGS.data_dir + "/汇总筛选/*.png", recursive=True)
        source_files += glob.glob(flags.FLAGS.data_dir + "/汇总筛选/*.jpg", recursive=True)
        source_files = [file for file in source_files if "bl" not in file]
        random.shuffle(source_files)
    else:
        source_files = glob.glob(flags.FLAGS.data_dir + "/val/**/*.png", recursive=True)


    # if flags.FLAGS.prefix == "train" or "train" in flags.FLAGS.prefix:
    #     source_files = glob.glob(flags.FLAGS.data_dir + "/*.jpg", recursive=True)
    #     new_source_files = []
    #     for file in source_files:
    #         if random.random() > 0.5:
    #             new_source_files.append(file)
    #     source_files = new_source_files
    #     random.shuffle(source_files)
    # else:
    #     source_files = glob.glob(flags.FLAGS.data_dir + "/val/**/*.png", recursive=True)

    print("total files", len(source_files))

    tfrecord_gen = tfrecord_writer.TFRecordGenerator(flags.FLAGS.target_dir, flags.FLAGS.prefix, 0, 2000)
    ds = tf.data.Dataset.from_tensor_slices(source_files)
    for file_path in ds:
        file_path = file_path.numpy().decode()
        if 'none' in file_path:
            source_img = tf.image.decode_image(tf.io.read_file(file_path), channels=3)
            split_and_save_2(tfrecord_gen, source_img)
        elif 'raw_data' in file_path or "images" in file_path or 'train_A' in file_path:
            source_img = tf.image.decode_image(tf.io.read_file(file_path), channels=3)
            if os.path.exists(file_path.replace('.png', '_bl.png')) is  True:
                tfile_path = file_path.replace('.png', '_bl.png')
            else:
                tfile_path = file_path.replace('.jpg', '_bl.jpg')
            # tfile_path = file_path.replace('train_A', 'train_B').replace('.jpg', '_gt.png')
            target_img = tf.image.decode_image(tf.io.read_file(tfile_path), channels=3)
            # split_and_save3(tfrecord_gen, file_path, tfile_path)
            split_and_save(tfrecord_gen, source_img, target_img)
        else:
            combined_img = tf.image.decode_image(tf.io.read_file(file_path), channels=3)
            source_img, _, target_img = tf.split(combined_img, 3, axis=1)

            if source_img.shape != target_img.shape:
                print(file_path)
                raise ValueError("src and target shape must match.")
            h, w, _ = source_img.shape
            if flags.FLAGS.prefix == "train":
                split_and_save(tfrecord_gen, source_img, target_img)
            else:
                source_img = tf.image.resize(source_img, (2208, 1664), method=tf.image.ResizeMethod.BILINEAR,
                                             preserve_aspect_ratio=False)
                target_img = tf.image.resize(target_img, (2208, 1664), method=tf.image.ResizeMethod.BILINEAR,
                                             preserve_aspect_ratio=False)
                tfrecord_gen.write_image_and_image(tf.image.encode_jpeg(tf.cast(source_img, tf.uint8)).numpy(),
                                                   tf.image.encode_png(tf.cast(target_img, tf.uint8)).numpy())

    print("total files", len(source_files))


if __name__ == '__main__':
    app.run(main)
