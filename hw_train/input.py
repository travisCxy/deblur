import tensorflow as tf
import horovod.tensorflow as hvd
import common
import glob, cv2, random, os
import numpy as np
from scipy.interpolate import UnivariateSpline
# from utils.utils_blindsr import add_blur, add_resize


def _create_LUT_8UC1(x, y):
  spl = UnivariateSpline(x, y)
  return spl(range(256))


class DataReader(object):

    def __init__(self, data_dir, is_training=True, do_random_crop=False, output_channels=1):
        self.is_training = is_training
        self.data_dir = data_dir
        self.do_random_crop = do_random_crop
        self.output_channels = output_channels

        self.incr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
        self.decr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

        self.table = None
        path = os.path.join(data_dir, 'score.txt')
        if os.path.exists(path):
            keys = []
            vals = []
            with open(path) as f:
                for line in f.readlines():
                    kv = line.strip().split(',')
                    keys.append(kv[0])
                    vals.append(float(kv[1]))
            init = tf.lookup.KeyValueTensorInitializer(keys, vals)
            self.table = tf.lookup.StaticHashTable(init, default_value=-1)

        ''' 
        c = 0
        for f in self._get_filenames():
            for record in tf.python_io.tf_record_iterator(f):
                c += 1
        print("total samples count %d" % c)
        '''

    def _get_filenames(self):
        if self.is_training:
            # return glob.glob(os.path.join(self.data_dir, "**/train*")) + glob.glob(os.path.join(self.data_dir, "train*"))
            return glob.glob(os.path.join(self.data_dir, "train*"))
        else:
            #return glob.glob(os.path.join(self.data_dir, "**/val*")) + glob.glob(os.path.join(self.data_dir, "val*"))
            return glob.glob(os.path.join(self.data_dir, "val*"))

    def get_dataset(self, batch_size):
        # def _parse_example_proto(example_serialized):
        #     feature_map = {
        #         'image/source': tf.io.FixedLenFeature([], dtype=tf.string,
        #                                               default_value=''),
        #         'image/target': tf.io.FixedLenFeature([], dtype=tf.string,
        #                                               default_value=''),
        #         # 'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,
        #         #                                       default_value=''),
        #     }
        #     features = tf.io.parse_single_example(
        #         serialized=example_serialized, features=feature_map)
        #
        #     return features['image/source'], features['image/target']
        def _parse_example_proto(example_serialized):
            feature_map = {
                'image/source': tf.io.FixedLenFeature([], dtype=tf.string,
                                                      default_value=''),
                'image/target': tf.io.FixedLenFeature([], dtype=tf.string,
                                                      default_value=''),
                'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
            }
            features = tf.io.parse_single_example(
                serialized=example_serialized, features=feature_map)

            return features['image/source'], features['image/target'], features['image/label']

        def _render_light(img, img2):
            c_r, c_g, c_b = cv2.split(img)
            c_r2, c_g2, c_b2 = cv2.split(img2)

            if random.random() > 0.5:
                if random.random() > 0.5:
                    for i in range(random.randint(0, 8)):
                        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
                        c_r2 = cv2.LUT(c_r2, self.incr_ch_lut).astype(np.uint8)
                if random.random() > 0.5:
                    for i in range(random.randint(0, 8)):
                        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
                        c_b2 = cv2.LUT(c_b2, self.decr_ch_lut).astype(np.uint8)
                img = cv2.merge((c_r, c_g, c_b))
                img2 = cv2.merge((c_r2, c_g2, c_b2))
            else:
                if random.random() > 0.5:
                    for i in range(random.randint(0, 8)):
                        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
                        c_r2 = cv2.LUT(c_r2, self.decr_ch_lut).astype(np.uint8)
                for i in range(random.randint(0, 8)):
                    c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
                    c_b2 = cv2.LUT(c_b2, self.incr_ch_lut).astype(np.uint8)
                img = cv2.merge((c_r, c_g, c_b))
                img2 = cv2.merge((c_r2, c_g2, c_b2))

            if random.random() > 0.5:
                c_h, c_s, c_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
                c_h2, c_s2, c_v2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_RGB2HSV))
                if random.random() > 0.5:
                    # decrease color saturation
                    c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
                    c_s2 = cv2.LUT(c_s2, self.decr_ch_lut).astype(np.uint8)
                else:
                    # increase color saturation
                    c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)
                    c_s2 = cv2.LUT(c_s2, self.incr_ch_lut).astype(np.uint8)
                img = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
                img2 = cv2.cvtColor(cv2.merge((c_h2, c_s2, c_v2)), cv2.COLOR_HSV2RGB)
            return img, img2

        def _render(img, img2):
            if random.random() < 0.1:
                img, img2 = _render_light(img, img2)
            if random.random() < 0.5:
                img, img2 = add_blur(img, img2, sf=random.choice([1,2,3,4]))
            return img, img2

        def _render_image(source_img, target_img):
            if self.is_training:
                src_shape = tf.shape(source_img)
                source_img, target_img = tf.numpy_function(_render, [source_img, target_img], [tf.uint8, tf.uint8])
                source_img = tf.reshape(source_img, src_shape)
                target_img = tf.reshape(target_img, src_shape)
            source_img = tf.cast(source_img, tf.float32)
            target_img = tf.cast(target_img, tf.float32)
            source_img = source_img / 127.5 - 1
            target_img = target_img / 255
            return source_img, target_img

        def _load_image_train(example_serialized):
            source_imgbuf, target_imgbuf, label = _parse_example_proto(example_serialized)

            if self.is_training and self.do_random_crop:
                source_img, target_img = common.random_crop_from_raw(source_imgbuf, target_imgbuf, 256)
            else:
                source_img = tf.image.decode_jpeg(source_imgbuf, 3)
                target_img = tf.image.decode_jpeg(target_imgbuf, self.output_channels)

            if self.is_training:
                source_img, target_img = common.preprocess(source_img, target_img, self.do_random_crop)
                if label == 0 and tf.random.uniform(()) < 0.1:
                    src_shape = tf.shape(source_img)
                    source_img, target_img = tf.numpy_function(_render_light, [source_img, target_img], [tf.uint8, tf.uint8])
                    source_img = tf.reshape(source_img, src_shape)
                    target_img = tf.reshape(target_img, src_shape)

            source_img = tf.cast(source_img, tf.float32)
            target_img = tf.cast(target_img, tf.float32)
            source_img = source_img / 127.5 - 1
            target_img = target_img / 255
            return source_img, target_img

        def _fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024  # 8 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        def _filter(example_serialized):
            # feature_map = {'image/filename': tf.io.FixedLenFeature([], tf.string, default_value='')}
            # features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
            # score = self.table.lookup(features['image/filename'])
            # return score > 0.7
            feature_map = {'image/source': tf.io.FixedLenFeature([], dtype=tf.string, default_value='')}
            features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
            shape = tf.image.extract_jpeg_shape(features['image/source'])[:2]
            return shape[0] > 256 and shape[1] > 256

        filenames = self._get_filenames()
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if self.is_training:
            dataset = dataset.shard(hvd.size(), hvd.rank())
            dataset = dataset.shuffle(buffer_size=len(filenames) // hvd.size())
            dataset = dataset.repeat(-1)
        else:
            dataset = dataset.repeat(1)
        # Read the data from disk in parallel
        dataset = dataset.apply(tf.data.experimental.parallel_interleave(
            _fetch_dataset, cycle_length=8, sloppy=True))
        if self.is_training:
            dataset = dataset.shuffle(1000)
            # dataset = dataset.filter(_filter)
        dataset = dataset.map(_load_image_train, num_parallel_calls=4 if self.is_training else 1)
        # dataset = dataset.map(_render_image, num_parallel_calls=8 if self.is_training else 1)
        dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    import os, cv2
    import numpy as np
    from tqdm import tqdm

    hvd.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    data_reader = DataReader("./datasets/train/20230428/tfrecords", False, True, 3)
    ds = data_reader.get_dataset(1)
    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", 800, 1000)
    count = 0
    for a, b in tqdm(ds):
        a = tf.cast((a + 1) * 127.5, tf.uint8)
        b = tf.cast(b * 255, tf.uint8)
        cv2.imshow('img', np.array(a[0])[:, :, ::-1])
        if cv2.waitKey(0) == 27:
            break
        cv2.imshow('img', np.array(b[0])[:, :, ::-1])
        if cv2.waitKey(0) == 27:
            break
        count += 1
    print(count)
