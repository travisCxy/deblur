import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'attention_decoder_op.so')
_attention_decoder_module = tf.load_op_library(filename)


# AttentionDecoderBlockCell
def attention_decoder_block_cell(x, a, keys, values, cs_prev, h_prev, lstm_w, lstm_b, query_w, attention_v,
                                 attention_w, project_w, project_b, finished_inputs, start_token, end_token, name=None):
    return _attention_decoder_module.attention_block_greedy_decoder_cell(keys, values, x, a, cs_prev,
              h_prev, lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b,
              finished_inputs,
              start_token=start_token, end_token=end_token)

def attention_block_beam_search_decoder_cell(x, a, keys, values, cs_prev, h_prev, lstm_w, lstm_b, query_w, attention_v,
                                 attention_w, project_w, project_b, finished_inputs, log_probs_inputs,
                                             start_token, end_token, beam_width, name=None):
    return _attention_decoder_module.attention_block_beam_search_decoder_cell(keys, values, x, a, cs_prev,
              h_prev, lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b,
              finished_inputs, log_probs_inputs,
              start_token=start_token, end_token=end_token, beam_width=beam_width)


def beam_search_gather(h_status, c_status, a_status, finished, beam_indices, beam_width):
    return _attention_decoder_module.beam_search_batch_gather(h_status, c_status, a_status, finished, beam_indices, beam_width)


def attention_block_greedy_decoder(x, a, keys, values, cs_prev, h_prev, lstm_w, lstm_b, query_w, attention_v,
                                 attention_w, project_w, project_b, finished_inputs, start_token, end_token, name=None):
    return _attention_decoder_module.attention_block_greedy_decoder(keys, values, x, a, cs_prev,
              h_prev, lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b,
                                                                    finished_inputs,
              start_token=start_token, end_token=end_token)

def attention_block_beam_search_decoder(x, a, keys, values, cs_prev, h_prev, lstm_w, lstm_b, query_w, attention_v,
                                 attention_w, project_w, project_b, finished_inputs, log_probs_inputs,
                                             start_token, end_token, beam_width, name=None):
    return _attention_decoder_module.attention_block_beam_search_decoder(keys, values, x, a, cs_prev,
              h_prev, lstm_w, lstm_b, query_w, attention_v, attention_w, project_w, project_b,
              finished_inputs, log_probs_inputs,
              start_token=start_token, end_token=end_token, beam_width=beam_width)