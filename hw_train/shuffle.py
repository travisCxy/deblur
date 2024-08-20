import multiprocessing
import random

from absl import flags, app
import tensorflow as tf
from glob import glob
import os, cv2
from tqdm import tqdm
import numpy as np
from utils import tfrecord_writer


flags.DEFINE_string('data_dir', "datasets", 'data dir.')
flags.DEFINE_string('target_dir', "datasets/tfrecords/20240528", 'target dir.')
flags.DEFINE_string('prefix', 'train', 'train/val.')
flags.DEFINE_integer('reader', 4, 'Reader process number.')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def read(image_queue, filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    for record in dataset:
        try:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            feature = example.features.feature
            features = {key:feature[key] for key in feature}
            image_queue.put(features)
        except Exception as ex:
            print(ex)


def write(k, image_queue, result_queue):
    tfrecord_gen = tfrecord_writer.TFRecordGenerator(flags.FLAGS.target_dir, flags.FLAGS.prefix, k, 2000)
    while True:
        features = image_queue.get()
        if features is None:
            break
        tfrecord_gen.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
        result_queue.put(1)


def main(_):
    files = glob(os.path.join(flags.FLAGS.data_dir, 'tfrecords/20231226/train*'))
    files += glob(os.path.join(flags.FLAGS.data_dir, 'train/20240524_by/tfrecords/*0528*'))
   # print(os.path.exists(os.path.join(flags.FLAGS.data_dir, 'train/20231020_水彩/tfrecords/*')))
    for file in files:
        print(file)
    random.shuffle(files)
    t = tqdm(total=len(files)*2000)
    files = np.array_split(files, flags.FLAGS.reader)
    image_queue = multiprocessing.Queue(flags.FLAGS.reader*20)
    file_readers = [multiprocessing.Process(target=read, args=(image_queue, files[i]))
                    for i in range(flags.FLAGS.reader)]
    for file_reader in file_readers:
        file_reader.start()
    result_queue = multiprocessing.Queue(60)
    runners = [multiprocessing.Process(target=write, args=(i, image_queue, result_queue))
               for i in range(flags.FLAGS.reader)]
    for runner in runners:
        runner.start()
    while True:
        if file_readers is not None:
            if not any(file_reader.is_alive() for file_reader in file_readers):
                file_readers = None
                for runner in runners:
                    image_queue.put(None)
        if runners is not None:
            if not any(runner.is_alive() for runner in runners):
                runners = None
                result_queue.put(None)
        try:
            ret = result_queue.get(timeout=1)
        except:
            continue
        if ret is None:
            break
        # t.set_postfix(score=ret)
        t.update(1)


if __name__ == '__main__':
    app.run(main)
