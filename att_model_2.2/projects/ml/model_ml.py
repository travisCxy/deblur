import tensorflow as tf
import input_utils

from projects.ml import common
from projects.ml import dataset

from kernels.tf_ctc_decoder_op.ctc_decoder_op import sample_id_decode

from decoder.att_decoder import AttenionDecoder
#from encoder.resenet_encoder import coordinate_embed_encoder
from encoder.mobilenet_encoder import coordinate_embed_encoder


def encoder():
    kwargs = {
        "model_type": 'large',
        "regularizer": None, # tf.keras.regularizers.l2(l=0.000004)
    }
    return coordinate_embed_encoder([dataset.MAX_HEIGHT, dataset.MAX_WIDTH, 3], **kwargs)


class MlModel(tf.keras.Model):
    def __init__(self):
        super(MlModel, self).__init__()
        self.encoder = encoder()
        self.decoder = AttenionDecoder(256, common.NUM_CLASSES, common.GO_SYMBOL, common.END_SYMBOL, common.PAD_SYMBOL, beam_width=3)

    def train(self, inputs):
        net = input_utils.image_process_for_train(inputs["images"])
        net = self.encoder(net, training=True)
        return self.decoder.train(net, inputs["labels"])

    def losses(self, logits, inputs):
        return self.decoder.main_loss(logits, inputs["labels"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, dataset.MAX_HEIGHT, dataset.MAX_WIDTH, 3], dtype=tf.uint8)])
    def infer(self, images):
        net = input_utils.image_process_for_val(images)
        net = self.encoder(net)
        #predicts = self.decoder.greedy_infer(net, int(common.MAX_SEQ_LEN))  # a little logger for frac.
        predicts = self.decoder.beam_search_infer(net, int(common.MAX_SEQ_LEN))
        codes = tf.constant(common.SYMBOLS, dtype=tf.string)
        ret = sample_id_decode(predicts, codes, common.GO_SYMBOL, common.END_SYMBOL, common.PAD_SYMBOL, 3)
        return {"strings": ret}

    def eval_infer(self, inputs):
        images = inputs["images"]
        return self.infer(images)

    def fake_infer(self):
        images = tf.zeros(shape=[1, dataset.MAX_HEIGHT, dataset.MAX_WIDTH, 3], dtype=tf.uint8)
        self.infer(images)


#def losses(logits, gt):
#    logits = tf.cast(logits, dtype=tf.float32)
#    gt_wo_go = gt[:, 1:]  # skip go sample
#    seq_len = tf.minimum(tf.shape(gt_wo_go)[1], tf.shape(logits)[1])
#    gt_wo_go = gt_wo_go[:, :seq_len]
#    logits = logits[:, :seq_len, :]
#    weights = tf.cast(tf.math.not_equal(gt_wo_go, common.PAD_SYMBOL), tf.float32)
#    train_loss = tfa.seq2seq.sequence_loss(logits, gt_wo_go, weights)
#    return train_loss


