import tensorflow as tf
import tensorflow_addons as tfa
import collections
from kernels.attention_decoder_op.attention_decoder import attention_decoder_block_cell, attention_block_greedy_decoder, \
    attention_block_beam_search_decoder_cell, attention_block_beam_search_decoder


class SimpleAttentionWrapperState(
    collections.namedtuple(
        "SimpleAttentionWrapperState",
        (
            "cell_state",
            'attention',
        ),
    )
):
    pass


class SimpleBeamSearchDecoderState(
    collections.namedtuple(
        "BeamSearchDecoderState",
        (
            "cell_state",
            "log_probs",
            "finished",
        ),
    )
):
    pass


class AttenionDecoder(tf.keras.Model):
    def __init__(self, num_units, num_outputs, go_symbol, end_symbol, pad_symbol, beam_width=2):
        super().__init__()
        self.num_units = num_units
        self.num_outputs = num_outputs
        self.go_symbol = go_symbol
        self.end_symbol = end_symbol
        self.pad_symbol = pad_symbol

        self.attention_mechanism = tfa.seq2seq.BahdanauAttention(self.num_units, memory=None)
        self.rnn_cell = tf.keras.layers.LSTMCell(self.num_units, implementation=2)
        #self.rnn_cell = LSTMBlockCell.LSTMBlockCell(self.num_units)
        self.decoder_cell = tfa.seq2seq.AttentionWrapper(
            self.rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.num_units,
            output_attention=True,
        )

        self.projection_layer = tf.keras.layers.Dense(self.num_outputs)

        train_sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.train_decoder = tfa.seq2seq.BasicDecoder(self.decoder_cell, train_sampler, output_layer=self.projection_layer)

        eval_sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler(embedding_fn=self.embed)
        self.eval_decoder = tfa.seq2seq.BasicDecoder(self.decoder_cell, eval_sampler, output_layer=self.projection_layer)

        self.beam_width = beam_width
        self.beam_decorder = tfa.seq2seq.BeamSearchDecoder(self.decoder_cell, beam_width=self.beam_width,
                                                           embedding_fn=self.embed, output_layer=self.projection_layer)

    def embed(self, labels):
        return tf.one_hot(labels, self.num_outputs, dtype=self._dtype_policy.compute_dtype)

    def train(self, inputs, labels):
        self.attention_mechanism.setup_memory(inputs)
        decoder_initial_state = self.decoder_cell.get_initial_state(inputs=inputs)
        label_embedings = self.embed(labels)

        pad_len = tf.reduce_sum(tf.cast(tf.equal(labels, self.pad_symbol), tf.int32), axis=-1)
        gt_lens = tf.shape(labels)[-1] - pad_len
        #print(label_embedings, decoder_initial_state)
        outputs, _, _ = self.train_decoder(
            label_embedings,
            initial_state=decoder_initial_state,
            sequence_length=gt_lens,
            training=True,
        )
        logits = outputs.rnn_output
        return logits

    def main_loss(self, logits, gt):
        logits = tf.cast(logits, dtype=tf.float32)
        gt_wo_go = gt[:, 1:]  # skip go sample
        seq_len = tf.minimum(tf.shape(gt_wo_go)[1], tf.shape(logits)[1])
        gt_wo_go = gt_wo_go[:, :seq_len]
        logits = logits[:, :seq_len, :]
        weights = tf.cast(tf.math.not_equal(gt_wo_go, self.pad_symbol), tf.float32)
        train_loss = tfa.seq2seq.sequence_loss(logits, gt_wo_go, weights)
        return train_loss

    def _prepare_weights(self, inputs, attention_state_shape):
        if not self.attention_mechanism.built:
            self.rnn_cell.build(tf.TensorShape([None, self.num_units + self.num_outputs]))

            rnn_out_shape = tf.TensorShape([None, self.num_units])

            memory_shape = self.attention_mechanism.values.shape
            self.attention_mechanism.build([rnn_out_shape, attention_state_shape, memory_shape])

            self.decoder_cell._attention_layers[0].build([None, inputs.get_shape()[2] + self.num_units])
            self.projection_layer.build([None, self.num_units])

        lstm_kernel = tf.concat([self.rnn_cell.kernel, self.rnn_cell.recurrent_kernel], axis=0)
        lstm_bias = self.rnn_cell.bias

        query_kernel = self.attention_mechanism.query_layer.kernel

        lstm_w = tf.cast(lstm_kernel, dtype=self._compute_dtype)
        lstm_b = tf.cast(lstm_bias, dtype=self._compute_dtype)
        query_w = tf.cast(query_kernel, dtype=self._compute_dtype)
        attention_v = tf.cast(self.attention_mechanism.attention_v, dtype=self._compute_dtype)
        attention_w = tf.cast(self.decoder_cell._attention_layers[0].kernel, dtype=self._compute_dtype)
        project_w = tf.cast(self.projection_layer.kernel, dtype=self._compute_dtype)
        project_b = tf.cast(self.projection_layer.bias, dtype=self._compute_dtype)
        return lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b

    def _block_greedy_decoder_cell(self, finsihed, inputs, attention, states, weights):
        (h_prev, cs_prev) = states
        (lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b) = weights

        (sample_ids, finished, next_inputs, project_out, next_lstm_cs,
         next_lstm_h, next_attention, alignments) \
            = attention_decoder_block_cell(
            keys=self.attention_mechanism.keys,
            values=self.attention_mechanism.values,
            x=inputs,
            a=attention,
            cs_prev=cs_prev,
            h_prev=h_prev,
            lstm_w=lstm_w,
            lstm_b=lstm_b,
            query_w=query_w,
            attention_v=attention_v,
            attention_w=attention_w,
            project_w=project_w,
            project_b=project_b,
            finished_inputs=finsihed,
            start_token=self.go_symbol,
            end_token=self.end_symbol,
        )
        return (next_lstm_h, [next_lstm_h, next_lstm_cs], alignments, next_attention,
               project_out, sample_ids, next_inputs, finished)

    def _greedy_decode_step(self, time, finished, inputs, state, weights):
        #pre_attention = state.attention
        #cell_state = state.cell_state
        #cell_inputs = tf.concat([inputs, state.attention], axis=-1)
        #cell_output, next_cell_state = self.rnn_cell(cell_inputs, cell_state)
        #query = cell_output
        #processed_query = self.attention_mechanism.query_layer(query)
        #processed_query = tf.expand_dims(processed_query, 1)
        #score = tf.reduce_sum(
        #    self.attention_mechanism.attention_v * tf.tanh(self.attention_mechanism.keys + processed_query),
        #    [2])
        #score = tf.reduce_sum(score_base, [2])
        #alignments = tf.nn.softmax(score, axis=-1)
        #next_attention_state = alignments

        #expanded_alignments = tf.expand_dims(alignments, 1)
        #context_ = tf.matmul(expanded_alignments, self.attention_mechanism.values)
        #context_ = tf.squeeze(context_, [1])

        #attention = attention_layer(tf.concat([cell_output, context_], 1))
        #attention = self.decoder_cell._attention_layers[0](tf.concat([cell_output, context_], 1))

        #if self.output_layer is not None:
        #    cell_outputs = self.output_layer(cell_outputs)
        #cell_outputs = self.projection_layer(attention)

        cell_state = state.cell_state
        (cell_output, next_cell_state, alignments, attention, project_out, sample_ids,
         next_inputs, finished) = self._block_greedy_decoder_cell(finished, inputs, state.attention, cell_state, weights)

        next_state = SimpleAttentionWrapperState(
            cell_state=next_cell_state,
            attention=attention,
        )

        cell_outputs = project_out
        cell_state = next_state
        next_state = cell_state
        outputs = tfa.seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
        return outputs, next_state, next_inputs, finished

    #@tf.autograph.experimental.do_not_convert()
    def greedy_infer(self, inputs, max_iterations):
        #self.decoder_cell._batch_size_checks = batch_size_checks

        self.attention_mechanism.setup_memory(inputs)

        batch = tf.shape(inputs)[0]

        start_tokens = tf.fill([batch], self.go_symbol)
        decoder_initial_state = self.decoder_cell.get_initial_state(inputs=inputs)
        attention_state_shape = decoder_initial_state.attention_state.shape

        decoder_initial_state = SimpleAttentionWrapperState(
            cell_state=decoder_initial_state.cell_state,
            attention=decoder_initial_state.attention,
        )

        first_finished, first_inputs, first_state = self.eval_decoder.initialize(None,
                                                                                 initial_state=decoder_initial_state,
                                                                                 start_tokens=start_tokens,
                                                                                 end_token=self.end_symbol)
        first_time = tf.constant(0, dtype=tf.int32)
        predictions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        weights = self._prepare_weights(inputs, attention_state_shape)
        loop_vars = [first_time, first_finished, first_inputs, first_state, predictions, weights]

        def condition(time, finished, inputs, state, output_array, weights):
            return tf.logical_and(time < max_iterations,  tf.logical_not(tf.reduce_all(finished)))

        def opt_body(time, finished, inputs, state, weights):
            outputs, next_state, next_inputs, step_finished = self._greedy_decode_step(time, finished, inputs, state, weights)
            #finished = tf.logical_or(finished, step_finished)
            finished = step_finished # block cell did this logic.
            return outputs, next_state, next_inputs, finished

        def body(time, finished, inputs, state, output_array, weights):
            outputs, next_state, next_inputs, finished = opt_body(time, finished, inputs, state, weights)
            output_array = output_array.write(time, outputs.sample_id)
            return time + 1, finished, next_inputs, next_state, output_array, weights

        _, _, _, _, predictions, _ = tf.while_loop(condition, body, loop_vars=loop_vars)
        return tf.transpose(predictions.stack(), [1, 0])

    def _block_greedy_decode(self, inputs, attention, states, weights):
        (h_prev, cs_prev) = states
        (lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b) = weights

        sample_ids = attention_block_greedy_decoder(
            keys=self.attention_mechanism.keys,
            values=self.attention_mechanism.values,
            x=inputs,
            a=attention,
            cs_prev=cs_prev,
            h_prev=h_prev,
            lstm_w=lstm_w,
            lstm_b=lstm_b,
            query_w=query_w,
            attention_v=attention_v,
            attention_w=attention_w,
            project_w=project_w,
            project_b=project_b,
            start_token=self.go_symbol,
            end_token=self.end_symbol,
        )
        return sample_ids

    def block_greedy_infer(self, inputs):
        self.attention_mechanism.setup_memory(inputs)

        batch = tf.shape(inputs)[0]
        start_tokens = tf.fill([batch], self.go_symbol)
        decoder_initial_state = self.decoder_cell.get_initial_state(inputs=inputs)
        attention_state_shape = decoder_initial_state.attention_state.shape

        decoder_initial_state = SimpleAttentionWrapperState(
            cell_state=decoder_initial_state.cell_state,
            attention=decoder_initial_state.attention,
        )

        first_finished, first_inputs, first_state = self.eval_decoder.initialize(None,
                                                                                 initial_state=decoder_initial_state,
                                                                                 start_tokens=start_tokens,
                                                                                 end_token=self.end_symbol)

        weights = self._prepare_weights(inputs, attention_state_shape)
        sample_ids = self._block_greedy_decode(first_inputs, decoder_initial_state.attention,
                                       decoder_initial_state.cell_state, weights)
        return tf.transpose(sample_ids, [1, 0])

    def _beam_decoder_step(self, time, inputs, state):
        # pre_attention = state.attention
        cell_state = state.cell_state.cell_state
        cell_inputs = tf.concat([inputs, state.cell_state.attention], axis=-1)
        cell_output, next_cell_state = self.rnn_cell(cell_inputs, cell_state)

        query = cell_output
        processed_query = self.attention_mechanism.query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1)
        score = tf.reduce_sum(
            self.attention_mechanism.attention_v * tf.tanh(self.attention_mechanism.keys + processed_query),
            [2])

        alignments = tf.nn.softmax(score, axis=-1)
        next_attention_state = alignments

        expanded_alignments = tf.expand_dims(alignments, 1)
        context_ = tf.matmul(expanded_alignments, self.attention_mechanism.values)
        context_ = tf.squeeze(context_, [1])

        # attention_layer is None?
        attention = self.decoder_cell._attention_layers[0](tf.concat([cell_output, context_], 1))

        #next_state = SimpleAttentionWrapperState(
        #    cell_state=next_cell_state,
        #    attention=attention,
        #)

        cell_outputs = self.projection_layer(attention)
        step_log_probs = tf.nn.log_softmax(cell_outputs)

        previously_finished = state.finished
        not_finished = tf.logical_not(previously_finished)
        finished_row = tf.one_hot(
            self.end_symbol,
            self.num_outputs,
            dtype=step_log_probs.dtype,
            on_value=tf.convert_to_tensor(0.0, dtype=step_log_probs.dtype),
            off_value=step_log_probs.dtype.min,
        )
        finished_probs = tf.tile(
            tf.reshape(finished_row, [1, -1]), tf.concat([tf.shape(previously_finished), [1]], 0)
        )
        finished_mask = tf.tile(tf.expand_dims(previously_finished, 1), [1, self.num_outputs])
        step_log_probs = tf.where(finished_mask, finished_probs, step_log_probs)
        total_probs = tf.expand_dims(state.log_probs, 1) + step_log_probs

        total_probs = tf.reshape(total_probs, [-1, self.beam_width * self.num_outputs])
        next_beam_scores, word_indices = tf.math.top_k(total_probs, k=self.beam_width)
        raw_next_word_ids = tf.math.floormod(
            word_indices, self.num_outputs, name="next_beam_word_ids"
        )
        next_word_ids = tf.cast(raw_next_word_ids, tf.int32)
        next_beam_ids = tf.cast(
            word_indices / self.num_outputs, tf.int32, name="next_beam_parent_ids"
        )
        next_word_ids = tf.reshape(next_word_ids, [-1])

        range_ = tf.expand_dims(tf.range(tf.shape(total_probs)[0]) * self.beam_width, 1)
        gather_indices = tf.reshape(range_ + next_beam_ids, [-1])
        next_beam_ids = tf.reshape(next_beam_ids, [-1])

        # not necessary since if it finished, then the prob will not change, it must be in top k.
        previously_finished = tf.gather(previously_finished, gather_indices)
        next_finished = tf.logical_or(
            previously_finished,
            tf.equal(next_word_ids, self.end_symbol),
            name="next_beam_finished",
        )
        next_cell_state[0] = tf.gather(next_cell_state[0], gather_indices)
        next_cell_state[1] = tf.gather(next_cell_state[1], gather_indices)
        attention = tf.gather(attention, gather_indices)

        next_state = SimpleAttentionWrapperState(
            cell_state=next_cell_state,
            attention=attention,
        )
        next_beam_scores = tf.reshape(next_beam_scores, [-1])
        next_state = SimpleBeamSearchDecoderState(next_state, next_beam_scores, next_finished)
        outputs = tfa.seq2seq.BeamSearchDecoderOutput(next_beam_scores, next_word_ids, next_beam_ids)
        next_inputs = self.embed(next_word_ids)
        return outputs, next_state, next_inputs, next_finished

    def _block_beam_search_decode_step(self, time, inputs, state, weights):

        (h_prev, cs_prev) = state.cell_state.cell_state
        (lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b) = weights

        attention = state.cell_state.attention
        finsihed = state.finished
        log_probs = state.log_probs
        (sample_ids, next_finished, next_inputs, project_out, next_lstm_cs,
         next_lstm_h, next_attention, beam_indices, next_log_probs, topk_indices) \
            = attention_block_beam_search_decoder_cell(
            keys=self.attention_mechanism.keys,
            values=self.attention_mechanism.values,
            x=inputs,
            a=attention,
            cs_prev=cs_prev,
            h_prev=h_prev,
            lstm_w=lstm_w,
            lstm_b=lstm_b,
            query_w=query_w,
            attention_v=attention_v,
            attention_w=attention_w,
            project_w=project_w,
            project_b=project_b,
            finished_inputs=finsihed,
            log_probs_inputs=log_probs,
            start_token=self.go_symbol,
            end_token=self.end_symbol,
            beam_width=self.beam_width,
        )

        next_beam_scores = next_log_probs
        next_word_ids = sample_ids
        next_beam_ids = beam_indices

        next_state = SimpleAttentionWrapperState(
            cell_state=[next_lstm_h, next_lstm_cs],
            attention=next_attention,
        )
        next_state = SimpleBeamSearchDecoderState(next_state, next_beam_scores, next_finished)
        outputs = tfa.seq2seq.BeamSearchDecoderOutput(next_beam_scores, next_word_ids, next_beam_ids)
        return outputs, next_state, next_inputs, next_finished

    def beam_search_infer(self, inputs, max_iterations):

        batch = tf.shape(inputs)[0]

        inputs = tfa.seq2seq.tile_batch(inputs, multiplier=self.beam_width)
        self.attention_mechanism.setup_memory(inputs)

        start_tokens = tf.fill([batch], self.go_symbol)

        decoder_initial_state = self.decoder_cell.get_initial_state(inputs=inputs)
        attention_state_shape = decoder_initial_state.attention_state.shape

        first_finished, first_inputs, first_state = self.beam_decorder.initialize(None,
                                                                                  initial_state=decoder_initial_state,
                                                                                  start_tokens=start_tokens,
                                                                                  end_token=self.end_symbol)

        simple_change = True
        if simple_change:
            first_state = SimpleBeamSearchDecoderState(
                cell_state=SimpleAttentionWrapperState(
                    cell_state=decoder_initial_state.cell_state,
                    attention=decoder_initial_state.attention,
                ),
                log_probs=tf.reshape(first_state.log_probs, [-1]),
                finished=tf.reshape(first_state.finished, [-1]),
            )
            first_finished = tf.reshape(first_finished, [-1])
            first_inputs = tf.reshape(first_inputs, [batch * self.beam_width, -1])

        first_time = tf.constant(0, dtype=tf.int32)
        parent_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        predict_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        weights = self._prepare_weights(inputs, attention_state_shape)
        loop_vars = [first_time, first_finished, first_inputs, first_state, [parent_ids, predict_ids], weights]

        def condition(time, finished, inputs, state, output_array, weights):
            return tf.logical_and(time < max_iterations,  tf.logical_not(tf.reduce_all(finished)))

        def opt_body(time, finished, inputs, state, weights):
            if simple_change:
                #outputs, next_state, next_inputs, finished = self._beam_decoder_step(time, inputs, state)
                outputs, next_state, next_inputs, step_finished = self._block_beam_search_decode_step(time, inputs, state, weights)
            else:
                outputs, next_state, next_inputs, step_finished = self.beam_decorder.step(time, inputs, state)

            return outputs, next_state, next_inputs, step_finished

        def body(time, finished, inputs, state, output_array, weights):
            outputs, next_state, next_inputs, finished = opt_body(time, finished, inputs, state, weights)
            output_array[0] = output_array[0].write(time, outputs.parent_ids)
            output_array[1] = output_array[1].write(time, outputs.predicted_ids)
            return time + 1, finished, next_inputs, next_state, output_array, weights

        _, _, _, _, (parent_ids, predict_ids), _ = tf.while_loop(condition, body, loop_vars=loop_vars)

        parent_ids = parent_ids.stack()
        predict_ids = predict_ids.stack()
        if len(parent_ids.shape) == 2:
            parent_ids = tf.reshape(parent_ids, [tf.shape(parent_ids)[0], -1, self.beam_width])
            predict_ids = tf.reshape(predict_ids, [tf.shape(parent_ids)[0], -1, self.beam_width])
        predicted_ids = tfa.seq2seq.gather_tree(
            predict_ids,
            parent_ids,
            max_sequence_lengths=tf.fill([batch], max_iterations),
            end_token=self.end_symbol,
        )
        return tf.transpose(predicted_ids, [1, 2, 0])

    def _block_beam_search_decode(self, inputs, state, weights):
        (lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b) = weights
        (h_prev, cs_prev) = state.cell_state.cell_state
        attention = state.cell_state.attention
        finsihed = state.finished
        log_probs = state.log_probs

        sample_ids, parent_ids = attention_block_beam_search_decoder(
            keys=self.attention_mechanism.keys,
            values=self.attention_mechanism.values,
            x=inputs,
            a=attention,
            cs_prev=cs_prev,
            h_prev=h_prev,
            lstm_w=lstm_w,
            lstm_b=lstm_b,
            query_w=query_w,
            attention_v=attention_v,
            attention_w=attention_w,
            project_w=project_w,
            project_b=project_b,
            finished_inputs=finsihed,
            log_probs_inputs=log_probs,
            start_token=self.go_symbol,
            end_token=self.end_symbol,
            beam_width=self.beam_width,
        )
        return sample_ids, parent_ids

    def block_beam_search_infer(self, inputs):
        #self.decoder_cell._batch_size_checks = batch_size_checks

        batch = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        inputs = tfa.seq2seq.tile_batch(inputs, multiplier=self.beam_width)
        self.attention_mechanism.setup_memory(inputs)

        start_tokens = tf.fill([batch], self.go_symbol)
        decoder_initial_state = self.decoder_cell.get_initial_state(inputs=inputs)
        attention_state_shape = decoder_initial_state.attention_state.shape

        first_finished, first_inputs, first_state = self.beam_decorder.initialize(None,
                                                                                  initial_state=decoder_initial_state,
                                                                                  start_tokens=start_tokens,
                                                                                  end_token=self.end_symbol)

        first_state = SimpleBeamSearchDecoderState(
            cell_state=SimpleAttentionWrapperState(
                cell_state=decoder_initial_state.cell_state,
                attention=decoder_initial_state.attention,
            ),
            log_probs=tf.reshape(first_state.log_probs, [-1]),
            finished=tf.reshape(first_state.finished, [-1]),
        )
        first_inputs = tf.reshape(first_inputs, [batch * self.beam_width, -1])
        weights = self._prepare_weights(inputs, attention_state_shape)
        predict_ids, parent_ids = self._block_beam_search_decode(first_inputs, first_state, weights)

        if len(predict_ids.shape) == 2:
            parent_ids = tf.reshape(parent_ids, [tf.shape(parent_ids)[0], -1, self.beam_width])
            predict_ids = tf.reshape(predict_ids, [tf.shape(parent_ids)[0], -1, self.beam_width])
        predicted_ids = tfa.seq2seq.gather_tree(
            predict_ids,
            parent_ids,
            max_sequence_lengths=tf.fill([batch], max_time),
            end_token=self.end_symbol,
        )
        return tf.transpose(predicted_ids, [1, 2, 0])

