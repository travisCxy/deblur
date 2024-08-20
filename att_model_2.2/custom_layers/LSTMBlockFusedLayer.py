import tensorflow as tf


class LSTMBlockFusedLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, forget_bias=1.0, cell_clip=None):
        super(LSTMBlockFusedLayer, self).__init__()  # 1080Ti fp32 > fp16, 2080Ti fp32 < fp16
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._cell_clip = cell_clip if cell_clip is not None else -1

    def build(self, input_shape):
        input_size = input_shape.dims[2].value
        self._kernel = self.add_variable(
            "kernel", [input_size + self._num_units, self._num_units * 4])
        self._bias = self.add_variable(
            "bias", [self._num_units * 4],
            initializer=tf.keras.initializers.constant(0.0))

    def call(self, inputs, **kwargs):
        inputs_shape = inputs.shape
        time_len, batch_size = inputs_shape[0], inputs_shape[1]
        if time_len is None:
            time_len = tf.shape(inputs)[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[1]

        #dtype = inputs.dtype
        z = tf.zeros(tf.stack([batch_size, self._num_units]), dtype=self._dtype_policy.compute_dtype)
        initial_state = z, z
        initial_cell_state, initial_output = initial_state
        max_seq_len = tf.cast(time_len, tf.int64)
        wci = wcf = wco = tf.zeros([self._num_units], dtype=self._dtype_policy.compute_dtype)
        _, cs, _, _, _, _, h = tf.raw_ops.BlockLSTM(
            seq_len_max=max_seq_len,
            x=inputs,
            cs_prev=initial_cell_state,
            h_prev=initial_output,
            w=self._kernel,
            wci=wci,
            wcf=wcf,
            wco=wco,
            b=self._bias,
            forget_bias=self._forget_bias,
            cell_clip=self._cell_clip,
            use_peephole=False)
        return h, cs