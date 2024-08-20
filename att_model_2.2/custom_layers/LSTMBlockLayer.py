import tensorflow as tf


class LSTMBlockLayer(tf.keras.layers.LSTM):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 time_major=False,
                 unroll=False,
                 **kwargs):
        super(LSTMBlockLayer, self).__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            time_major=time_major,
            unroll=unroll,
            **kwargs
        )
        # a bug is here. do not use gpu in tf <= 2.2
        self._could_use_gpu_kernel = False

    def call1(self, inputs):
        #keras IFCO, however LSTMBlockCell ICFO
        k_i, k_f, k_c, k_o = tf.split(
            self.cell.kernel, num_or_size_splits=4, axis=1)

        rk_i, rk_f, rk_c, rk_o = tf.split(
            self.cell.recurrent_kernel, num_or_size_splits=4, axis=1)

        b_i, b_f, b_c, b_o = tf.split(
            self.cell.bias, num_or_size_splits=4, axis=0)

        kernel = tf.concat([tf.concat([k_i, k_c, k_f, k_o], axis=1),
                           tf.concat([rk_i, rk_c, rk_f, rk_o], axis=1)], axis=0)
        bias = tf.concat([b_i, b_c, b_f, b_o], axis=0)

        inputs_shape = inputs.shape
        time_len, batch_size = inputs_shape[0], inputs_shape[1]
        if time_len is None:
            time_len = tf.shape(inputs)[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[1]

        z = tf.zeros(tf.stack([batch_size, self.cell.units]), dtype=self._dtype_policy.compute_dtype)
        initial_state = z, z
        initial_cell_state, initial_output = initial_state

        max_seq_len = tf.cast(time_len, tf.int64)
        wci = wcf = wco = tf.zeros([self.cell.units], dtype=self._dtype_policy.compute_dtype)
        #tf.raw_ops.BlockLSTMV2
        _, cs, _, _, _, _, h = tf.raw_ops.BlockLSTM(
            seq_len_max=max_seq_len,
            x=inputs,
            cs_prev=initial_cell_state,
            h_prev=initial_output,
            w=kernel,
            wci=wci,
            wcf=wcf,
            wco=wco,
            b=bias,
            forget_bias=0.0,
            cell_clip=-1,
            use_peephole=False)
        return h, cs

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if training:
            return super(LSTMBlockLayer, self).call(inputs, mask=mask, training=training, initial_state=initial_state)
        else:
            if initial_state is not None:
                raise NotImplemented()
            #inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)
            h, cs = self.call1(inputs)
            if not self.return_sequences or not self.return_state:
                raise NotImplemented()
            return h, None, None # not supported
