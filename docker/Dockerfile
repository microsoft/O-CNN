FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libcgal-dev \
        libeigen3-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-setuptools \
        python-scipy \
        rsync \
        software-properties-common \
        vim \
        zip && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
        rm get-pip.py

ENV WORKSPACE=/workspace
ENV PATH="${WORKSPACE}/bin:${PATH}"
WORKDIR $WORKSPACE/bin

ENV OCNN_ROOT=$WORKSPACE/ocnn
WORKDIR $OCNN_ROOT

ARG OCNN_COMMIT=origin/master
RUN git clone https://github.com/Microsoft/O-CNN.git . && git reset --hard $OCNN_COMMIT

WORKDIR $OCNN_ROOT/octree/Octree
RUN c++ main.cpp Octree.cpp  -O3 -DNDEBUG --std=c++11 -fPIC -Wall -Wno-sign-compare -Wno-uninitialized -fopenmp -o octree && \
cp octree $WORKSPACE/bin

WORKDIR $OCNN_ROOT/octree/Octree2PLY
RUN c++ Octree2Ply.cpp  -O3 -DNDEBUG --std=c++11 -fPIC -Wall -Wno-sign-compare -Wno-uninitialized -fopenmp -o octree2ply && \
cp octree2ply $WORKSPACE/bin

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

ARG CAFFE_COMMIT=6bfc5ca8f7c2a4b7de09dfe7a01cf9d3470d22b3
RUN git clone https://github.com/BVLC/caffe.git . && git reset --hard $CAFFE_COMMIT && \
    rsync -a $OCNN_ROOT/caffe/ ./ && ex -sc '83i|set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")' -cx CMakeLists.txt && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)" && \
    mkdir $CAFFE_ROOT/include/caffe/proto && \
    cp $CAFFE_ROOT/build/include/caffe/proto/caffe.pb.h $CAFFE_ROOT/include/caffe/proto/caffe.pb.h

ENV OCNN_TOOLS_ROOT=$WORKSPACE/ocnn_tools
WORKDIR $OCNN_TOOLS_ROOT

ARG OCNN_TOOLS_COMMIT=bbcf7e28e8500024a000b6b15eecd24652d6e11d
RUN git clone https://github.com/wang-ps/O-CNN.git . && git reset --hard $OCNN_TOOLS_COMMIT

WORKDIR $OCNN_TOOLS_ROOT/virtual_scanner/VirtualScanner
RUN c++ main.cpp VirtualScanner.cpp  -O3 -DNDEBUG --std=c++11 -fPIC -Wall -Wno-sign-compare -Wno-uninitialized -fopenmp -lboost_filesystem -lboost_system -lCGAL -I /usr/include/eigen3 -o virtualscanner && \
cp virtualscanner $WORKSPACE/bin

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
