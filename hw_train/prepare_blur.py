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
flags.DEFINE_string('data_dir', "datasets/train/20231226_by/raw_data", 'data dir.')
flags.DEFINE_string('target_dir', "datasets/train/20240126_blur/tfrecords", 'target dir.')
flags.DEFINE_string('prefix', 'val-deblur', 'train/val.')
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
    # if flags.FLAGS.prefix == "train" or "train" in flags.FLAGS.prefix:
    #     # source_files = glob.glob(flags.FLAGS.data_dir + "/train/20231113_水彩/raw_data3/*_crop.png", recursive=True)
    #     source_files = glob.glob(flags.FLAGS.data_dir + "/*/*_crop.png", recursive=True)
    #     source_files = [file for file in source_files if "gt" not in file]
    #     random.shuffle(source_files)
    # else:
    #     source_files = glob.glob(flags.FLAGS.data_dir + "/val/**/*.png", recursive=True)
    # print("total files", len(source_files))

    fin = open("/mnt/server_data/code/projects/deblur/data/val_1221.txt", "r")
    lines = fin.readlines()
    source_files = []
    target_files = {}
    for line in lines:
        line = line.strip().split()
        source_files.append(line[0])
        target_files[line[0]] = line[1]
    tfrecord_gen = tfrecord_writer.TFRecordGenerator(flags.FLAGS.target_dir, flags.FLAGS.prefix, 0, 2000)
    ds = tf.data.Dataset.from_tensor_slices(source_files)
    for file_path in ds:
        file_path = file_path.numpy().decode()

        source_img = tf.image.decode_image(tf.io.read_file(file_path), channels=3)
        # if os.path.exists(file_path.replace('_crop.png', '_crop_bl.png')) is  True:
        #     tfile_path = file_path.replace('_crop.png', '_crop_bl.png')
        # else:
        #     tfile_path = file_path.replace('_crop.png', '_crop_gt.png')
        tfile_path = target_files[file_path]
        print(file_path)
        print(tfile_path)
        target_img = tf.image.decode_image(tf.io.read_file(tfile_path), channels=3)
        split_and_save(tfrecord_gen, source_img, target_img)


    print("total files", len(source_files))


if __name__ == '__main__':
    app.run(main)
