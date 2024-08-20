import tensorflow as tf
from absl import flags, app
import pdb
import time
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import horovod.tensorflow as hvd
import importlib

flags.DEFINE_string('model_dir', "./models_sl", 'model dir.')
flags.DEFINE_string('data_dir', "/mnt/data2/hwhelper_train/data/sl/tfrecords_old/*", 'model dir.')
flags.DEFINE_string('project', "sl", """train project""")

FLAGS = flags.FLAGS


def get_l2_loss(model, weight_decay=1e-4):
    filter_vars = []
    for v in model.trainable_variables:
        if "conv" in v.name and "kernel" in v.name:
            filter_vars.append(v)
    return weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, dtype=tf.float32)) for v in filter_vars])


@tf.function
def train_step(model, inputs, optimizer, weight_decay):
    loss_scale = 32.0
    with tf.GradientTape() as tape:
        logits = model.train(inputs)
        model_loss = model.losses(logits, inputs)

        l2_loss = get_l2_loss(model, weight_decay)
        total_loss = model_loss + l2_loss
        total_loss = total_loss * loss_scale

    tape = hvd.DistributedGradientTape(tape)

    trainable_variables = model.trainable_variables
    gradients_1 = tape.gradient(total_loss, trainable_variables)
    gradients = [tf.clip_by_value(grad / loss_scale, -5.0, 5.0) for grad in gradients_1]
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"model_loss": model_loss, "l2_loss": l2_loss, "total_loss": model_loss + l2_loss}


def config_optimizer(train_config):
    if train_config["decay_name"] == "manual":
        steps = [x // hvd.size() for x in train_config["decay_steps"]]
        lrs = [x * hvd.size() for x in train_config["lr"]]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(steps, lrs)

    if train_config["opt"] == "adam":
        optimizer = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)
        return optimizer




def main(_):
    if hvd.rank() == 0:
        summary_writer = tf.summary.create_file_writer("./logs")

    print("============ load project: %s ================"%FLAGS.project)
    project = importlib.import_module("projects." + FLAGS.project)
    configs = project.get_model_configs()

    data_path = FLAGS.data_dir
    dataset = configs["input_fn"](data_path, True)

    train_config = configs["train_config"]
    optimizer = config_optimizer(train_config)
    weight_decay = 1e-4
    if "weight_decay" in configs:
        weight_decay = configs["weight_decay"]
    model = configs["model"]
    global_step = tf.Variable(0, dtype=tf.int64)
    checkpoint_dir = FLAGS.model_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,  model=model, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=200)
    # last_checkpoint = '/mnt/server_data/data/sequni/save_model_old/ckpt-378001'
    # h5_weights = '/mnt/server_data/data/sequni/save_model_old/best_model.h5'
    # model.built = True
    # model.load_weights(h5_weights)

    last_checkpoint = ckpt_manager.latest_checkpoint
    if last_checkpoint is not None:
        print("loading from existing checkpoint " + last_checkpoint)
        checkpoint.restore(last_checkpoint)
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)
        hvd.broadcast_variables([global_step], root_rank=0)
        global_step = tf.Variable(84001, dtype=tf.int64)
        print("steps: ", global_step.numpy())
        print("loading from existing checkpoint done")

    start = time.time()
    for inputs in dataset:
        losses = train_step(model, inputs, optimizer, weight_decay)
        global_step.assign_add(1)
        steps = global_step.numpy()

        if steps == 1:
            hvd.broadcast_variables(model.variables, root_rank=0)

            if hvd.rank() == 0:
                tensorboard = tf.keras.callbacks.TensorBoard(checkpoint_dir, write_graph=True)
                tensorboard.set_model(model)

        if steps % 100 == 1 and hvd.local_rank() == 0:
            end = time.time()
            msg = ""
            for loss_k, loss_v in losses.items():
                msg += "" + loss_k + ": " + str(loss_v.numpy()) + " "
            print(msg, "time:", end - start, "step:", steps, "lr:", optimizer._decayed_lr(tf.float32).numpy())
            start = end

            tf.summary.experimental.set_step(steps)
            with summary_writer.as_default():
                for k, v in losses.items():
                    tf.summary.scalar(k, v)
                tf.summary.scalar("learning_rate", optimizer._decayed_lr(tf.float32))

                for var in model.trainable_variables:
                    tf.summary.histogram(var.name, var)

        if steps % 1000 == 1 and hvd.rank() == 0:
            ckpt_manager.save(checkpoint_number=steps)
            print("check point saved===================================")

        if steps >= train_config["max_step"] // hvd.size():
            ckpt_manager.save(checkpoint_number=steps)
            break


if __name__ == '__main__':
    import os
    #os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    app.run(main)
    #r = reader("/mnt/data2/hwhelper_train/data/seqch/tfrecords_gen/*", True)
    #r.prepare_inputs(1)
    #dataset = r.get_inputs_for_clone()
    #for items in dataset:
    #    print(items)
