import tensorflow as tf
from projects.hw import common
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tqdm
from absl import flags, app
import itertools
import importlib
from tensorflow.keras.utils import plot_model
flags.DEFINE_string('checkpoint_file', "", 'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_string('val_data_path', "", 'model dir.')
# flags.DEFINE_string('project', "ml", """train project""")
flags.DEFINE_string('project', "uni", """train project""")
# flags.DEFINE_string('project', "cht", """train project""")

FLAGS = flags.FLAGS







def eval(_):
    # checkpoint_path = "/mnt/server_data/data/sequni/models_equ_new/ckpt-44001"
    # reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    # checkpoint_path = "/mnt/server_data/data/sequni/save_model_old/ckpt-378001"



    test_cht = False
    checkpoint_path = FLAGS.checkpoint_file
    data_path = FLAGS.val_data_path

    # test_cht = True
    # checkpoint_path = "./save_model_cht/ckpt-272001"
    checkpoint_path = "/mnt/server_data/data/sequni/models_equ_cht/ckpt-350001"
    # checkpoint_path = "/mnt/server_data/data/sequni/models_equ_new/ckpt-82001"
    data_path = "/mnt/server_data/data/sequni/tfrecords_full/val*"
    print(FLAGS.project)


    project = importlib.import_module("projects." + FLAGS.project)
    configs = project.get_model_configs()

    dataset = configs["input_fn"](data_path, False)

    model = configs["model"]
    evaluator = configs["evaluator"]

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save('ckpt')
    checkpoint.restore(checkpoint_path).expect_partial()

    ind = 0
    for inputs in tqdm.tqdm(dataset, total=19200):
        ind += 1
        outputs = model.eval_infer(inputs)
        #print(outputs)
        if ind >500:
           #tf.profiler.experimental.stop()
           break
        evaluator.record_step(outputs, inputs, test_cht=test_cht)
    evaluator.eval_finished(checkpoint_path)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    app.run(eval)
