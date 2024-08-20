import tensorflow as tf
from absl import flags, app
import model
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# flags.DEFINE_string('checkpoint_path', "model_dir/240524/best.ckpt",
#                     'checkpoint path.')
# flags.DEFINE_string('export_dir', "model_dir/240524/saved_model", 'eval only.')
#

flags.DEFINE_string('checkpoint_path', "../hw_pp_train/model_dir/240529/ckpt-300000",
                    'checkpoint path.')
flags.DEFINE_string('export_dir', "../hw_pp_train/model_dir/240529/saved_model1", 'eval only.')

def export_trt(saved_model):
    print('Converting to TF-TRT FP16...')
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=(1 << 32))
    conversion_params = conversion_params._replace(precision_mode="FP16")


    conversion_params = conversion_params._replace(
        minimum_segment_size=20)
    conversion_params = conversion_params._replace(
        maximum_cached_engines=100)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model, conversion_params=conversion_params)
    converter.convert()

    converter.save(output_saved_model_dir='saved_model_trt16')
    print('Done Converting to TF-TRT FP16')


def export_saved_model():
    if os.path.exists(flags.FLAGS.export_dir):
        shutil.rmtree(flags.FLAGS.export_dir)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    unet = model.Generator_resnet(output_channels=3)
    # unet = model.Generator_efficientnetv2(output_channels=3)
    inputs = tf.keras.layers.Input(shape=[None, None, 3], dtype=tf.uint8, name="input_2")
    norm_inputs = tf.cast(inputs, tf.float16)
    norm_inputs = norm_inputs / 127.5 - 1
    outputs = unet(norm_inputs)
    outputs = tf.clip_by_value(outputs, 0, 1)
    outputs = tf.cast(outputs * 255, tf.uint8)
    infer_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    checkpoint = tf.train.Checkpoint(model=unet)
    status = checkpoint.restore(flags.FLAGS.checkpoint_path)
    status.expect_partial()
    #infer_model.save(flags.FLAGS.export_dir, save_format="tf")
    tf.saved_model.save(infer_model,flags.FLAGS.export_dir)
    print("exported %s to %s" % (flags.FLAGS.checkpoint_path, flags.FLAGS.export_dir))


def main(_):
    export_saved_model()
    #export_trt(flags.FLAGS.export_dir)


if __name__ == '__main__':
    app.run(main)
