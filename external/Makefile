SHELL=/bin/bash

nvcc=/usr/local/cuda-9.0/bin/nvcc
cudalib=/usr/local/cuda-9.0/lib64
TF_CFLAGS=-I/usr/local/lib/python3.5/dist-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0
TF_LFLAGS=-L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework
$(info $$TF_CFLAGS is ${TF_CFLAGS})
$(info $$TF_LFLAGS is ${TF_LFLAGS})


all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC ${TF_CFLAGS} -lcudart ${TF_LFLAGS} -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu ${TF_CFLAGS} ${TF_LFLAGS} -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC ${TF_CFLAGS} -lcudart ${TF_LFLAGS} -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu ${TF_CFLAGS} ${TF_LFLAGS} -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o
