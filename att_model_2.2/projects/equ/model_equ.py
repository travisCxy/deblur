import tensorflow as tf
import input_utils

from encoder.resenet_encoder import encoder as encoder_fn
from decoder.ctc_decoder import CtcDecoder

from projects.equ import common
from projects.equ import dataset


def encoder():
    kwargs = {
        "depth_rate": 1.0 / 4,
        "min_depth": 16,
        "blocks": [1, 2, 5, 3],
        "for_att": False,
        "data_format": "channels_last"
    }
    return encoder_fn([dataset.MAX_HEIGHT, None, 3], True, **kwargs)


class EquModel(tf.keras.Model):
    def __init__(self):
        super(EquModel, self).__init__()
        self.encoder = encoder()
        self.main_decoder = CtcDecoder(512, common.NUM_CLASSES, True, common.NUM_CLASSES-1)
        self.num_decoder = CtcDecoder(512, common.NUM_CLASSES, False, common.NUM_CLASSES-1)
        self.den_decoder = CtcDecoder(512, common.NUM_CLASSES, False, common.NUM_CLASSES-1)

    def train(self, inputs):
        net = input_utils.image_process_for_train(inputs["images"])
        net = self.encoder(net, training=True)
        main_logits = self.main_decoder.train(net)
        num_logits = self.num_decoder.train(net)
        den_logits = self.den_decoder.train(net)
        return [main_logits, num_logits, den_logits]

    def losses(self, logits, inputs):
        main_loss = self.main_decoder.main_loss(logits[0], inputs["labels"][:, 0, :], inputs["seq_lens"])
        num_loss = self.num_decoder.main_loss(logits[1], inputs["labels"][:, 1, :], inputs["seq_lens"])
        den_loss = self.den_decoder.main_loss(logits[2], inputs["labels"][:, 2, :], inputs["seq_lens"])
        return main_loss + num_loss + den_loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, dataset.MAX_HEIGHT, None, 3], dtype=tf.uint8),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def infer(self, images, seq_lens):
        images = input_utils.image_process_for_val(images)
        codes = tf.constant(common.DIGITS, dtype=tf.string)
        net = self.encoder(images)
        main_ret, _ = self.main_decoder.beam_infer(net, seq_lens, codes, 5)
        num_ret, _ = self.num_decoder.beam_infer(net, seq_lens, codes, 2)
        den_ret, _ = self.den_decoder.beam_infer(net, seq_lens, codes, 2)

        main_strings = tf.identity(main_ret[0], name="main_strings")
        main_ref_inds = tf.identity(main_ret[1], name="main_ref_inds")
        main_confs = tf.identity(main_ret[2], name="main_confs")

        num_strings = tf.identity(num_ret[0], name="num_strings")
        num_ref_inds = tf.identity(num_ret[1], name="num_ref_inds")
        num_confs = tf.identity(num_ret[2], name="num_confs")

        den_strings = tf.identity(den_ret[0], name="den_strings")
        den_ref_inds = tf.identity(den_ret[1], name="den_ref_inds")
        den_confs = tf.identity(den_ret[2], name="den_confs")

        return {
            "main_strings": main_strings,
            "main_ref_inds": main_ref_inds,
            "main_confs": main_confs,
            "num_strings": num_strings,
            "num_ref_inds": num_ref_inds,
            "num_confs": num_confs,
            "den_strings": den_strings,
            "den_ref_inds": den_ref_inds,
            "den_confs": den_confs,
        }

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, dataset.MAX_HEIGHT, None, 3], dtype=tf.uint8),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def infer2(self, images, seq_lens):
        images = input_utils.image_process_for_val(images)
        codes = tf.constant(common.DIGITS, dtype=tf.string)
        net = self.encoder(images)
        main_ret, main_logits = self.main_decoder.beam_infer(net, seq_lens, codes, 5)

        greedy_inds = tf.argmax(main_logits, axis=-1)
        has_frac = tf.reduce_any(tf.equal(greedy_inds, common.code2vec(common.FRAC_SYMBOL)[0]))

        main_strings = tf.identity(main_ret[0], name="main_strings")
        main_ref_inds = tf.identity(main_ret[1], name="main_ref_inds")
        main_confs = tf.identity(main_ret[2], name="main_confs")

        def true_fn():
            num_ret, _ = self.num_decoder.beam_infer(net, seq_lens, codes, 2)
            den_ret, _ = self.den_decoder.beam_infer(net, seq_lens, codes, 2)

            num_strings = tf.identity(num_ret[0], name="num_strings")
            num_ref_inds = tf.identity(num_ret[1], name="num_ref_inds")
            num_confs = tf.identity(num_ret[2], name="num_confs")

            den_strings = tf.identity(den_ret[0], name="den_strings")
            den_ref_inds = tf.identity(den_ret[1], name="den_ref_inds")
            den_confs = tf.identity(den_ret[2], name="den_confs")
            return (num_strings, num_ref_inds, num_confs), (den_strings, den_ref_inds, den_confs)

        def false_fn():
            batch_size = tf.shape(main_logits)[0]

            strings = tf.constant([['~', '~']], dtype=tf.string)
            ref_inds = tf.constant([1], dtype=tf.int32)
            confs = tf.constant([0.0], dtype=tf.float32)

            strings = tf.tile(strings, multiples=[batch_size, 1])
            ref_inds = tf.tile(ref_inds, [batch_size])
            confs = tf.tile(confs, [batch_size])
            return (strings, ref_inds, confs), (strings, ref_inds, confs)

        nums, dens = tf.cond(has_frac, true_fn, false_fn)

        return {
            "main_strings": main_strings,
            "main_ref_inds": main_ref_inds,
            "main_confs": main_confs,
            "num_strings": nums[0],
            "num_ref_inds": nums[1],
            "num_confs": nums[2],
            "den_strings": dens[0],
            "den_ref_inds": dens[1],
            "den_confs": dens[2],
        }

    def eval_infer(self, inputs):
        images = inputs["images"]
        seq_lens = inputs["seq_lens"]

        images = tf.reshape(images, shape=[-1, dataset.MAX_HEIGHT, dataset.MAX_WIDTH, 3])
        seq_lens = tf.reshape(seq_lens, [-1])
        return self.infer(images, seq_lens)

    def fake_infer(self):
        images = tf.zeros(shape=[1, dataset.MAX_HEIGHT, dataset.MAX_WIDTH, 3], dtype=tf.uint8)
        seq_lens = tf.constant([10], dtype=tf.int32)
        self.infer(images, seq_lens)
