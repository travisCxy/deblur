import tensorflow as tf
import numpy as np
import time
import ctc_decoder_op
from yyt import common

data = np.load("data/logits.npy")
data = data[:1,::]
seq_lens = [64] * data.shape[0]
data_pl = tf.placeholder(tf.float32, [None,None,None])
ret_tensor,ref_ind_tensor,conf_tensor = ctc_decoder_op.ctc_decode(data_pl, seq_lens, common.STD_DIGITS, 5)
std_label_coder = common.LabelCoder(common.STD_DIGITS, common.STD_LABEL_LEN, common.STD_OFFSET)
with tf.train.MonitoredSession() as sess:  
    s = time.time()
    for i in range(1):
        ret,ref_inds,confs = sess.run([ret_tensor,ref_ind_tensor,conf_tensor],feed_dict={data_pl:data})
    e = time.time()
    print(ret)
    print((e-s))
    
 

 
