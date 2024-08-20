import tensorflow as tf
import pdb
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 定义解析 TFRecord 的函数
def _parse_example_proto(example_serialized):
    feature_map = {
        'image/source': tf.io.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
        'image/target': tf.io.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,default_value=''),
        'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }
    features = tf.io.parse_single_example(
        serialized=example_serialized, features=feature_map)

    return features['image/source'], features['image/target'], features['image/label'], features['image/filename']

# 读取 TFRecord 文件，并解析数据
record_files = []
root_dir = './datasets/tfrecords/20240528'
for file in os.listdir(root_dir):
    if 'val-' in file:
        record_files.append(os.path.join(root_dir, file))
dataset = tf.data.TFRecordDataset(record_files)
dataset = dataset.map(_parse_example_proto)


# output_dir = 'raw_data'
output_dir = '/mnt/server_data2/code/shijuanbao_models/hw_train/datasets/raw_data/20240528/val'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#label_file = open(os.path.join(output_dir, 'label.txt'), 'w')
# 打印数据
index = 1
for data in dataset:
    source, target, label, filename = data
    label = tf.cast(label, tf.int32).numpy()
    filename = tf.cast(filename, tf.string).numpy().decode()
    source_img_name = os.path.join(output_dir, 'img_%d.jpg' % index)
    target_img_name = os.path.join(output_dir, 'img_%d_gt.png' % index)
    if 'deg' in filename:
        source_img_name = os.path.join(output_dir, 'img_%d_deg.jpg' % index)
        target_img_name = os.path.join(output_dir, 'img_%d_deg_gt.png' % index)
    tf.io.write_file(source_img_name, source)
    tf.io.write_file(target_img_name, target)
    #label_file.write(source_img_name + '\t' + target_img_name + '\t' + str(label) + '\t' + filename + '\n')
    print(index, filename)
    #pdb.set_trace()
    index += 1
# else:
#     source_img = tf.io.decode_jpeg(source)
#     target_img = tf.io.decode_png(target)
#     cv2.imwrite(source_img_name, source_img.numpy())
#     cv2.imwrite(target_img_name, target_img.numpy())