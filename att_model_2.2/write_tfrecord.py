#write tfrecord
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tqdm


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(imagepath, label):
    img = cv2.imread(imagepath)
    imgname = os.path.basename(imagepath)
    _, img_bytes = cv2.imencode(".jpg", img)
    img_str = img_bytes.tostring()
    feature = {
        'image/encoded': _bytes_feature(img_str),
        'image/format': _bytes_feature('jpeg'.encode('utf-8')),
        'image/label': _bytes_feature(label.encode('utf-8')),
        'image/name': _bytes_feature(imgname.encode('utf-8')),
        'image/cls': _int64_feature(1)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_one_record(lines, tfrecord_file):
    with  tf.io.TFRecordWriter(tfrecord_file) as writer:
        for line in tqdm.tqdm(lines, total=len(lines)):
            line = line.strip().split('\t')
            imagepath = line[0]
            label = line[-1]
            imagepath = os.path.join('/home/ateam/xychen/dataset/ocr/local_dataset/text_reg/taigi_images/cht_100w', imagepath)
            if not os.path.exists(imagepath):
                continue
            example = serialize_example(imagepath, label)
            writer.write(example)


def write_tfrecord(train_list_file, tfrecord_file):
    with open(train_list_file, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 5000):
        create_one_record(lines[i:i+5000], tfrecord_file.replace('.tfrecord', '_%d.tfrecord'%i))


write_tfrecord('/home/ateam/xychen/dataset/ocr/local_dataset/text_reg/taigi_images/cht_100w/train_label2.txt', '/mnt/server_data/data/sequni/tfrecords_full/train_cht_100w.tfrecord')
# write_tfrecord('/home/ateam/xychen/dataset/ocr/local_dataset/text_reg/taigi_images/cht/train_label.txt', '/mnt/server_data/data/sequni/tfrecords_cht/train_cht.tfrecord')
# write_tfrecord('/mnt/server_data/code/PaddleOCR_new/output/word_cht_test_data_preds.txt', '/mnt/server_data/data/sequni/tfrecords_cht/val_cht.tfrecord')