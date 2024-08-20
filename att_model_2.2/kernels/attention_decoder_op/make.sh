#!/usr/bin/env bash
TF_CFLAGS=$(python3.8 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python3.8 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CUDA_HOME=/usr/local/cuda
echo $TF_CFLAGS
echo $TF_LFLAGS

/usr/local/cuda/bin/nvcc -O3 -std c++11 -c -x cu -o attention_decoder_gpu.cu.o attention_decoder_gpu.cu.cc \
	-I . -I /usr/local/ -D GOOGLE_CUDA=1 -D NDEBUG -D THRUST_IGNORE_CUB_VERSION_CHECK $TF_CFLAGS \
	--expt-relaxed-constexpr -Xcompiler -fPIC

/usr/local/cuda/bin/nvcc -O3 -std c++11 -c -x cu -o reduction_ops_gpu_float.cu.o reduction_ops_gpu_float.cu.cc \
	-I . -I /usr/local/ -D GOOGLE_CUDA=1 -D NDEBUG -D THRUST_IGNORE_CUB_VERSION_CHECK $TF_CFLAGS \
	--expt-relaxed-constexpr -Xcompiler -fPIC

g++ -O2 -std=c++11 -shared  -D GOOGLE_CUDA=1 -D THRUST_IGNORE_CUB_VERSION_CHECK  -I $CUDA_HOME/include -o attention_decoder_op.so attention_decoder_reg.cc \
  attention_decoder.cc blas_gemm.cc attention_decoder_gpu.cu.o reduction_ops_gpu_float.cu.o $TF_CFLAGS $TF_LFLAGS -fPIC
