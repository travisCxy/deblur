import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'ctc_decoder.so')
_ctc_decoder_module = tf.load_op_library(filename)
ctc_decode = _ctc_decoder_module.ctc_decode
 
sample_id_decode = _ctc_decoder_module.sample_id_decode

ctc_greedy_decode = _ctc_decoder_module.ctc_greedy_decode