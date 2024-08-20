import os
import tensorflow as tf
import six


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label, synset):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(synset),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class TFRecordGenerator(object):
    def __init__(self, target_dir, prefix, prefix_index=0, capacity_per_tfrecord=5000):
        self.target_dir = target_dir
        self.prefix = prefix
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        self.prefix_index = prefix_index
        self.cur_index = 0
        self.total_count = 0
        self.capacity_per_tfrecord = capacity_per_tfrecord
        self.writer = None

    def _get_writer(self):
        if self.writer is None:
            self.writer = tf.io.TFRecordWriter(
                "%s/%s-%02d-%05d" % (self.target_dir, self.prefix, self.prefix_index, self.cur_index))
        elif self.total_count % self.capacity_per_tfrecord == 0:
            self.cur_index += 1
            self.writer.close()
            self.writer = tf.io.TFRecordWriter(
                "%s/%s-%02d-%05d" % (self.target_dir, self.prefix, self.prefix_index, self.cur_index))
            print("new writer %s/%s-%02d-%05d" % (self.target_dir, self.prefix, self.prefix_index, self.cur_index))
        return self.writer

    def write(self, example):
        writer = self._get_writer()
        writer.write(example)
        self.total_count += 1

    def write_image_and_label(self, source_image, target_image, label=0):
        if label == 0:
            self.write_image_and_image(source_image, target_image)
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/source': _bytes_feature(source_image),
                'image/target': _bytes_feature(target_image),
                'image/label': _int64_feature(label)}))
            self.write(example.SerializeToString())

    def write_image_and_mask(self, source_image, target_image, mask, filename, label=0):
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/source': _bytes_feature(source_image),
            'image/target': _bytes_feature(target_image),
            'image/mask': _bytes_feature(mask),
            'image/filename': _bytes_feature(filename),
            'image/label': _int64_feature(label)}))
        self.write(example.SerializeToString())

    def write_image_and_image(self, source_image, target_image):
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/source': _bytes_feature(source_image),
            'image/target': _bytes_feature(target_image)}))
        self.write(example.SerializeToString())

    def write_image_and_masks(self, source_image, body_mask, edge_masks):
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/source': _bytes_feature(source_image),
            'image/body_mask': _bytes_feature(body_mask),
            'image/edge_mask_1': _bytes_feature(edge_masks[0]),
            'image/edge_mask_2': _bytes_feature(edge_masks[1]),
            'image/edge_mask_3': _bytes_feature(edge_masks[2]),
            'image/edge_mask_4': _bytes_feature(edge_masks[3]),
        }
        ))
        self.write(example.SerializeToString())

    def close(self):
        if self.writer is not None:
            self.writer.close()
