import tensorflow as tf
import input_utils

from encoder.inception_encoder import encoder
from decoder.ctc_decoder import CtcDecoder

from projects.ch import common
from projects.ch import dataset


class CHModel(tf.keras.Model):
    def __init__(self):
        super(CHModel, self).__init__()
        self.encoder = encoder([dataset.MAX_HEIGHT, None, 3])
        self.decoder = CtcDecoder(256, common.NUM_CLASSES, True, common.NUM_CLASSES-1)

    def train(self, inputs):
        net = input_utils.image_process_for_train(inputs["images"])
        net = self.encoder(net, training=True)
        return self.decoder.train(net)

    def losses(self, logits, inputs):
        return self.decoder.main_loss(logits, inputs["labels"], inputs["seq_lens"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, dataset.MAX_HEIGHT, None, 3], dtype=tf.uint8),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def infer(self, images, seq_lens):
        images = input_utils.image_process_for_val(images)
        codes = tf.constant(common.DIGITS, dtype=tf.string)
        net = self.encoder(images)
        ret = self.decoder.greedy_infer(net, seq_lens, codes)
        return {"strings": tf.identity(ret, name="strings")}

    def eval_infer(self, inputs):
        images = inputs["images"]
        seq_lens = inputs["seq_lens"]
        return self.infer(images, seq_lens)

    def fake_infer(self):
        images = tf.zeros(shape=[1, dataset.MAX_HEIGHT, dataset.MAX_WIDTH, 3], dtype=tf.uint8)
        seq_lens = tf.constant([10], dtype=tf.int32)
        self.infer(images, seq_lens)