import tensorflow as tf
import input
import model
import time
import glob
import shutil
import horovod.tensorflow as hvd
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import vgg_style_loss
#import vgg_style_loss19 as vgg_style_loss
import eval
import argparse
import logging
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=3, help='batch size')
parser.add_argument('--output_channels', type=int, default=3, help='output channels')
parser.add_argument('--train_steps', type=int, default=200000, help='train steps')
parser.add_argument('--data_dir', type=str, default='./datasets/tfrecords', help='data dir')
parser.add_argument('--model_dir', type=str, default='./model_dir', help='model dir')
parser.add_argument('--eval_metric', type=str, default='iou', help='l1 or iou')
parser.add_argument('--eval_only', action='store_true', help='eval only')
parser.add_argument('--vgg_loss', action='store_true', help='if vgg loss applied')
parser.add_argument('--random_crop', action='store_true', help='do random crop or not')
parser.add_argument('--mixed_precision', action='store_true', help='mixed precision')
parser.add_argument('--model_name', type=str, default='resnet34', help='model name')
args = parser.parse_args()
best_score = 0


def get_model():
    if args.model_name == "mobilenet":
        return model.Generator(args.output_channels)
    if args.model_name == "resnet34":
        return model.Generator_resnet(args.output_channels)
    if args.model_name == "efficientnetv2":
        return model.Generator_efficientnetv2(args.output_channels)
    return None




def do_eval(checkpoint_path, unet=None,save_results=False):
    global best_score

    logging.info("evaluate " + checkpoint_path + " start.")
    if unet is None:
        unet = get_model()
        checkpoint = tf.train.Checkpoint(model=unet)
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
    if tf.io.gfile.isdir(args.model_dir):
        eval_output_dir = args.model_dir + "/eval_results"
    else:
        eval_output_dir = os.path.dirname(args.model_dir) + "/eval_results"

    if not os.path.exists(eval_output_dir):
        os.mkdir(eval_output_dir)
    accuracy = eval.eval_from_model(args.data_dir, unet, args.output_channels,format="ckpt",save_result_dir=eval_output_dir if save_results else None)
    if not args.eval_only and accuracy > best_score:
        logging.info("saving better result of " + checkpoint_path)
        for fpath in glob.glob("%s*" % (checkpoint_path)):
            step = int(checkpoint_path.split("-")[-1])
            shutil.copy(fpath, fpath.replace("ckpt-%d" % step, "best.ckpt"))
        out_fp = open(os.path.join(args.model_dir, "best.txt"), "w")
        out_fp.write("%s %.3f\n" % (checkpoint_path, accuracy))
        out_fp.close()
        best_score = accuracy
    return accuracy


def gradient_map_tf(x):
    batch_size, h_x, w_x, channel = x.shape
    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    r = padded[:, 1:h_x +1, 2:, :]
    l = padded[:, 1:h_x +1, :w_x , :]
    t = padded[:, 2:, 1:w_x + 1, :]
    b = padded[:, :h_x , 1:w_x + 1, :]
    xgrad = tf.pow(tf.pow((r - l) * 0.5, 2) + tf.pow((t - b) * 0.5, 2) + 1e-6, 0.5)
    return xgrad

def gradient_profile_loss(x,y):
    g_x = gradient_map_tf(x)
    g_y = gradient_map_tf(y)
    return tf.reduce_mean(tf.abs(g_x - g_y))

@tf.function
def training_step(unet_model, style_model, opt, images, labels, first_batch):
    with tf.GradientTape() as tape:
        gen_output = unet_model(images, training=True)
        gen_output = tf.cast(gen_output, tf.float32)
        if args.vgg_loss:
            gen_style = style_model(gen_output)
            target_style = style_model(labels)
            style_loss, content_loss = vgg_style_loss.style_content_loss(gen_style, target_style)
            tv_loss = tf.reduce_mean(tf.image.total_variation(gen_output)) * 0.01
        else:
            style_loss = 0
            content_loss = 0
            tv_loss = 0
        l1_loss = tf.reduce_mean(tf.abs(labels - gen_output)) * 100
        l2_loss = 0
        gp_loss = 0
        #l2_loss = tf.reduce_mean(tf.nn.l2_loss(labels - gen_output))
        #gp_loss = gradient_profile_loss(labels,gen_output) * 50
        total_loss = l1_loss + l2_loss + content_loss + tv_loss + gp_loss
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(total_loss, unet_model.trainable_variables)
    opt.apply_gradients(zip(grads, unet_model.trainable_variables))
    if first_batch:
        hvd.broadcast_variables(unet_model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return {"l1_loss": l1_loss,  "style_loss": style_loss, "content_loss": content_loss,
            "tv_loss": tv_loss,"l2_loss":l2_loss,
            "total_loss": total_loss}, gen_output


def main():
    logging.basicConfig(filename=os.path.join(args.model_dir, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('gpu: ' + str(hvd.rank()) + ' ' + str(args))
    logging.info(glob.glob(os.path.join(args.data_dir, "**/train*"))+glob.glob(os.path.join(args.data_dir, "train*")))
    logging.info(glob.glob(os.path.join(args.data_dir, "**/val*"))+glob.glob(os.path.join(args.data_dir, "val*")))

    if args.mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    if args.eval_only:

        if tf.io.gfile.isdir(args.model_dir):
            do_eval(tf.train.latest_checkpoint(args.model_dir),save_results=True)
        else:
            do_eval(args.model_dir,save_results=True)
        return

    unet = get_model()
    unet.summary()
    global_step = tf.Variable(0, dtype=tf.int64)
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [args.train_steps * 2 // 3, args.train_steps * 7 // 8],
        [1e-4 * hvd.size(), 1e-5 * hvd.size(), 1e-6 * hvd.size()])
    opt = tf.keras.optimizers.Adam(lr)
    #opt = tfa.optimizers.AdamW(learning_rate=lr,weight_decay=1e-4)
    if args.mixed_precision:
        opt = mixed_precision.LossScaleOptimizer(opt, loss_scale=256)
    checkpoint = tf.train.Checkpoint(model=unet, optimizer=opt, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, directory=args.model_dir, max_to_keep=5)
    last_checkpoint = ckpt_manager.latest_checkpoint
    # last_checkpoint = './model_dir/0414/ckpt-400000'
    if last_checkpoint is not None:
        logging.info("loading from existing checkpoint " + last_checkpoint)
        checkpoint.restore(last_checkpoint)
        hvd.broadcast_variables(unet.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
        hvd.broadcast_variables([global_step], root_rank=0)
    train_dataset = input.DataReader(args.data_dir, True, args.random_crop,
                                     args.output_channels).get_dataset(
        args.batch_size)
    if hvd.rank() == 0:
        summary_writer = tf.summary.create_file_writer(args.model_dir)
    if args.vgg_loss:
        style_model = vgg_style_loss.StyleContentModel()
    else:
        style_model = None
    start_time = time.time()
    for images, labels in train_dataset:
        global_step.assign_add(1)
        steps = global_step.numpy()
        loss_dict, gen_output = training_step(unet, style_model, opt, images, labels, steps == 1)
        if hvd.rank() != 0:
            continue
        # print logs every 10 steps

        if steps % 10 == 0:
            loss_str = ""
            for k, v in loss_dict.items():
                loss_str += k + ":" + str(v.numpy()) + " "
            samples_per_sec = int(10 * args.batch_size / (time.time() - start_time))
            logging.info('Step #%d\t%s\tsamples per sec %d' % (steps, loss_str, samples_per_sec))

            start_time = time.time()
        # save summary every 100 steps
        if steps % 100 == 0:
            tf.summary.experimental.set_step(steps)
            with summary_writer.as_default():
                tf.summary.scalar("learning_rate", lr(steps))
                for k, v in loss_dict.items():
                    tf.summary.scalar(k, v)
                for var in unet.trainable_variables:
                    tf.summary.histogram(var.name, var)
                # tf.summary.image("input_images", (images + 1) / 2, max_outputs=5)
                # tf.summary.image("outputs", gen_output, max_outputs=5)
        # save checkpoints every 1000 steps
        if steps % 5000 == 0:
            ckpt_manager.save(checkpoint_number=steps)
            with summary_writer.as_default():
                score = do_eval(ckpt_manager.latest_checkpoint, unet)
                tf.summary.scalar("MeanAbsoluteError", score)
        if steps >= args.train_steps:
            logging.info("train done.")
            break


if __name__ == '__main__':
    hvd.init()
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    main()
