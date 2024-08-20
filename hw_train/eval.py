import tensorflow as tf
import glob
import model
import os
import math
import cv2
import input
import numpy as np
import time
import logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision
DIFF_THRESHOLD = 5

def get_input_dataset(data_dir, output_channels):
    return input.DataReader(data_dir, False, False, output_channels).get_dataset(1)


def load_model_from_ckpt(ckpt_path, output_channels=3):
    unet = model.Generator_resnet(output_channels)
    checkpoint = tf.train.Checkpoint(model=unet)
    status = checkpoint.restore(ckpt_path)
    status.expect_partial()
    return unet


def load_model_from_saved_model(saved_model_path, output_channels=3):
    loaded_model = tf.saved_model.load(saved_model_path)
    return loaded_model


@tf.function
def step(model, inputs):
    inputs = tf.cast(inputs, tf.float16)
    masks = model(inputs, training=False)
    masks = tf.clip_by_value(masks, 0, 1)
    masks = tf.cast(masks * 255, tf.int32)
    return masks


def make_diff_and_save(source_img, target_img, target_file_path):
    img_gray = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
    diff = np.abs(img_gray - img2_gray) > DIFF_THRESHOLD

    diff_img = (diff * 255).astype(np.uint8)
    diff_img = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2RGB)
    final_img = np.concatenate([source_img, diff_img, target_img], axis=1)
    cv2.imwrite(target_file_path[:-4]+'_0331.png', final_img[:, :, ::-1])


def eval_from_model(data_dir, unet, output_channels, format="ckpt", save_result_dir=None):
    test_dataset = get_input_dataset(data_dir, output_channels)
    acc_sum = 0
    total_count = 0
    start_time = time.time()
    summary_images = []
    for batch, (images, labels) in enumerate(test_dataset):
        if format == "ckpt":
            masks = step(unet, images)
            if batch < 10:
                summary_images.append(tf.cast(masks, tf.uint8))
        else:
            inputs = tf.cast((images + 1) * 127.5, tf.uint8)
            masks = unet.signatures["serving_default"](inputs)
            masks = list(masks.values())[0]
            masks = tf.cast(masks, tf.int32)
        # cv2.imwrite("/mnt/data1/datasets/hwremove/output/%d.png" % batch,
        #            tf.cast(masks, tf.uint8).numpy()[0][..., ::-1])
        #weights = tf.cast(images < 0, tf.int32) + 1
        labels = tf.cast(labels * 255, tf.int32)
        source_imgs = tf.cast((images + 1) * 127.5, tf.int32)
        weights = tf.cast(tf.math.abs(source_imgs - labels) >= DIFF_THRESHOLD, tf.int32)
        cls_masks = tf.cast(tf.math.abs(masks - labels) <= DIFF_THRESHOLD, tf.int32)
        _, h, w, _ = cls_masks.shape
        acc = tf.reduce_sum(cls_masks * weights) / tf.reduce_sum(weights)
        acc_sum += acc
        #print(acc)
        total_count += 1
        if save_result_dir is not None:
            source_img = tf.cast((images + 1) * 127.5, tf.uint8).numpy()[0]
            target_img = masks.numpy().astype(np.uint8)[0]
            make_diff_and_save(source_img, target_img, "%s/%d.png" % (save_result_dir, total_count))
        if batch > 0 and batch % 10 == 0:
            logging.info("evaluate steps %d" % batch)
        #if batch >= 20:
        #    break
    if format == "ckpt":
        tf.summary.image("eval_outputs", tf.concat(summary_images, axis=0), max_outputs=10)
    end_time = time.time()
    accuracy = acc_sum / total_count
    logging.info("evaluate complete in %.2f seconds, with mean accuracy %.4f" % (end_time - start_time, accuracy))
    return accuracy


def eval_once(data_dir, model_path, output_channels, format="ckpt",save_result_dir=None):
    if format == "ckpt":
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        unet = load_model_from_ckpt(model_path, output_channels)
    else:
        unet = load_model_from_saved_model(model_path, output_channels)
    eval_from_model(data_dir, unet, output_channels, format,save_result_dir)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    data_dir = "./datasets/tfrecords/20230911"
    output_dir = "./model_dir/230921/eval_results"
    ckpt_path = "./model_dir/230921/best.ckpt"
    # saved_model_path = "/mnt/ceph/shijuanbao_deploy/shijuanbao_engine/triton/models/hw_model/20210126/model.savedmodel"
    saved_model_path = "./model_dir/0331/saved_model"

    #saved_model_path = "/mnt/cephfs/datasets/hwremove/saved_model"
    # eval_once(data_dir, ckpt_path, 3)
    eval_once(data_dir, ckpt_path, 3, "ckpt", output_dir)
    '''
    data_dir = "/mnt/cephfs/datasets/prettyprint/tfrecords"
    saved_model_path = "/mnt/data1/experiments/shijuanbao_engine_git/models/pp_model"
    # eval_once(data_dir, ckpt_path, 3)
    eval_once(data_dir, saved_model_path, 1, "saved_model")
    '''
