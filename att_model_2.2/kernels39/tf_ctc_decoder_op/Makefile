TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_LIB = /usr/local/cuda/lib64

all: clean build test

build: roi_pooling.so

roi_pooling.cu.o: roi_pooling.cu.cc
	/usr/local/cuda/bin/nvcc -std=c++11 -c -o $@ $? -I $(TF_INC) -I $(TF_INC)/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D _GLIBCXX_USE_CXX11_ABI=0

roi_pooling.so: roi_pooling.cc roi_pooling.cu.o
	g++ -std=c++11 -shared -o $@ $? -I $(TF_INC) -I $(TF_INC)/external/nsync/public -fPIC -lcudart -ltensorflow_framework -L/usr/local/lib/python2.7/dist-packages/tensorflow -L$(CUDA_LIB) -D _GLIBCXX_USE_CXX11_ABI=0

test: build
	python roi_pooling_test.py

clean:
	rm -f *.o *.so *.pyc *.npy
