#!/usr/bin/env bash
TF_CFLAGS=$(python3.8 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python3.8 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

echo $TF_CFLAGS
echo $TF_LFLAGS

g++ -O2 -std=c++11 -shared  -o ctc_decoder.so ctc_decoder_op.cc  $TF_CFLAGS $TF_LFLAGS -fPIC
