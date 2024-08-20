# -*-coding:utf-8-*-
import os
import threading
import lmdb
import tensorflow as tf
from queue import Queue
import glob


def parse_example(example):
    features = tf.io.parse_single_example(example,
                                          features={
                                              'image/name': tf.io.FixedLenFeature([], tf.string),
                                              'image/label': tf.io.FixedLenFeature([], tf.string),
                                              'image/encoded': tf.io.FixedLenFeature([], tf.string),
                                              'image/cls': tf.io.FixedLenFeature([], tf.int64)
                                          })
    cls = features['image/cls']
    label = features['image/label']
    #image = tf.image.decode_jpeg(features['image/encoded'])
    image_bytes = features['image/encoded']
    filename = features['image/name']

    return image_bytes, label, cls, filename

def read_tfrecord(data_path_partten):
    dataset = tf.data.Dataset.list_files(data_path_partten, shuffle=False)
    dataset = dataset.repeat(1)
    dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename),
                                 cycle_length=1)
    dataset = dataset.map(parse_example)
    return dataset


# 定义一个函数，该函数从队列中获取TFRecord文件名并处理
def worker(input_queue, output_dir, worker_id=1):
    env = lmdb.open(output_dir, map_size=int(1e12), max_dbs=0)
    count = 0
    with env.begin(write=True) as txn:
        while True:
            file = input_queue.get()
            print(file)
            if file is None or input_queue.empty():
                break
            data = read_tfrecord(file)

        # 解析TFRecord文件
            for i, (image, label, cls, filename) in enumerate(data):
                cls = cls.numpy()
                label = label.numpy()
                image = image.numpy()
                # 构造键值对，这里我们使用一个简单的计数器作为键
                image_key = "img_%09d" % count
                label_key = "label_%09d" % count
                cls_key = "cls_%09d" % count
                txn.put(image_key.encode('ascii'), image)
                txn.put(label_key.encode('ascii'), label)
                txn.put(cls_key.encode('ascii'), cls)
                count += 1
                if count % 1000 == 0:
                    print(f'Processed {count} records')

            input_queue.task_done()
        txn.put(b'num_samples', str(count).encode('ascii'))

def read_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, map_size=int(1e12), max_dbs=0)
    stats = env.stat()
    num_entries = stats['entries']
    print(f'Found {num_entries} entries in LMDB')
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            print(key)


# 输入文件列表
tfrecord_files = glob.glob('/mnt/server_data2/data/seq_chemical/tfrecords_20240712_latex/val*')

# 输出目录
output_dir = './lmdb_output'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)



# 创建队列和线程
input_queue = Queue()
num_workers = 5  # 可以根据系统资源调整线程数量
workers = []

# 启动线程
# for i in range(num_workers):
#     # = threading.Thread(target=worker, args=(input_queue, output_dir))
#     t = threading.Thread(target=worker, args=(input_queue, output_dir, i))
#     t.start()
#     workers.append(t)

#填充队列
for file in tfrecord_files:
    input_queue.put(file)


worker(input_queue, output_dir)
#read_lmdb(output_dir)
# 等待所有任务完成
#input_queue.join()

# 结束线程
# for _ in range(num_workers):
#     input_queue.put(None)
# for t in workers:
#     t.join()

print("转换完成！")