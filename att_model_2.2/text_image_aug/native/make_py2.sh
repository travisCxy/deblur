
OPENCV_FLAGS=`pkg-config --libs opencv`
g++ -O2 -shared -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -o Augment.so  src/Augment.cpp src/conversion.cpp  src/imgwarp_mls.cpp  src/imgwarp_mls_similarity.cpp  -I ./include -I /usr/include -I /usr/include/python2.7 -I /usr/local/lib/python2.7/dist-packages/numpy/core/include -I /usr/local/include -L /usr/lib/x86_64-linux-gnu/ -l boost_python-py27 $OPENCV_FLAGS
