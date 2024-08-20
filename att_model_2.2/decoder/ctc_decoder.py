import tensorflow as tf
from custom_layers.LSTMBlockFusedLayer import LSTMBlockFusedLayer
from custom_layers.LSTMBlockLayer import LSTMBlockLayer
from kernels.tf_ctc_decoder_op.ctc_decoder_op import ctc_decode, sample_id_decode, ctc_greedy_decode


def decoder(num_units, num_output, dtype):
    # inputs: [time_len, batch_size, channels]

    inputs = tf.keras.layers.Input([None, 768], dtype=dtype)

    net_fw, _ = LSTMBlockFusedLayer(num_units)(inputs)

    inputs_bw = tf.reverse(inputs, axis=[0])
    net_bw, _ = LSTMBlockFusedLayer(num_units)(inputs_bw)
    net_bw = tf.reverse(net_bw, axis=[0])

    net = tf.concat([net_fw, net_bw], axis=2)

    net = tf.keras.layers.Conv1D(num_output, 1, data_format="channels_last",
                                 kernel_initializer=tf.initializers.VarianceScaling(scale=2.0))(net)
    return tf.keras.Model(inputs=inputs, outputs=net, name="decoder")


class CtcDecoder(tf.keras.Model):
    def __init__(self, num_units, num_outputs, use_bilstm, blank_symbol):
        super().__init__()
        self.num_units = num_units
        self.num_outputs = num_outputs
        self.use_bilstm = use_bilstm
        self.blank_symbol = blank_symbol

        self.fw_lstm = LSTMBlockLayer(self.num_units, return_sequences=True, return_state=True)
        if self.use_bilstm:
            self.bw_lstm = LSTMBlockLayer(self.num_units, return_sequences=True, return_state=True)

        self.project = tf.keras.layers.Conv1D(self.num_outputs, 1,  data_format="channels_last",
                                              kernel_initializer=tf.initializers.VarianceScaling(scale=2.0))

    def calc_logits(self, inputs, training):
        net, _, _ = self.fw_lstm(inputs, training=training)
        if self.use_bilstm:
            inputs_bw = tf.reverse(inputs, axis=[0])
            net_bw, _, _ = self.bw_lstm(inputs_bw, training=training)
            net_bw = tf.reverse(net_bw, axis=[0])

            net = tf.concat([net, net_bw], axis=2)
        logits = self.project(net)
        return logits

    def train(self, inputs):
        return self.calc_logits(inputs, training=True)

    def main_loss(self, logits, gt, seq_lens):
        logits = tf.cast(logits, dtype=tf.float32)
        batch_size = gt.shape[0]
        indices = tf.where(tf.not_equal(gt, -1))
        sparse_gt = tf.SparseTensor(indices, tf.gather_nd(gt, indices), gt.get_shape())
        ctc_loss = tf.nn.ctc_loss(sparse_gt, logits, seq_lens, [tf.shape(logits)[0]] * batch_size,
                                  logits_time_major=True, blank_index=self.blank_symbol)
        return tf.reduce_mean(ctc_loss)

    def greedy_infer(self, inputs, seq_lens, codes):
        logits = self.calc_logits(inputs, training=False)
        greedy_inds = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
        greedy_inds = tf.transpose(greedy_inds, [1, 0])
        ret = ctc_greedy_decode(greedy_inds, seq_lens, codes, self.blank_symbol)
        return tf.expand_dims(ret, axis=1)

    def beam_infer(self, inputs, seq_lens, codes, beam_width):
        logits = self.calc_logits(inputs, training=False)
        logits = tf.transpose(logits, (1, 0, 2))
        logits = tf.cast(logits, dtype=tf.float32)
        return ctc_decode(logits, seq_lens, codes, beam_width), logits