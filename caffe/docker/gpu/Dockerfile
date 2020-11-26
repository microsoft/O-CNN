FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# dir
ENV WORKSPACE=/workspace
ENV OCNN_ROOT=$WORKSPACE/ocnn
ENV CAFFE_ROOT=$WORKSPACE/caffe


# dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
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
        python-pip \
        python-setuptools \
        python-scipy \
        rsync \
        vim \
        zip && \
    rm -rf /var/lib/apt/lists/*


# cmake
WORKDIR $WORKSPACE
RUN wget https://cmake.org/files/v3.16/cmake-3.16.2-Linux-x86_64.sh && \
    mkdir cmake-3.16.2 && \
    sh cmake-3.16.2-Linux-x86_64.sh --prefix=$WORKSPACE/cmake-3.16.2 --skip-license && \
    ln -s $WORKSPACE/cmake-3.16.2/bin/cmake /usr/bin/cmake && \
    rm cmake-3.16.2-Linux-x86_64.sh


# nccl
WORKDIR $WORKSPACE
ARG NCCL_COMMIT=286916a1a37ca1fe8cd43e280f5c42ec29569fc5
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    git reset --hard $NCCL_COMMIT && \
    make -j install && \
    cd .. && \
    rm -rf nccl


# ocnn
WORKDIR $OCNN_ROOT
RUN git clone https://github.com/Microsoft/O-CNN.git .
RUN cd octree/external && \
    git clone --recursive https://github.com/wang-ps/octree-ext.git && \
    cd .. && \
    mkdir build && \
    cd build && \
    cmake ..  && \
    cmake --build . --config Release


# caffe
WORKDIR $CAFFE_ROOT
ARG CAFFE_COMMIT=6bfc5ca8f7c2a4b7de09dfe7a01cf9d3470d22b3
RUN git clone https://github.com/BVLC/caffe.git . && \
    git reset --hard $CAFFE_COMMIT && \
    rsync -a $OCNN_ROOT/caffe/ ./ && \
    pip install --upgrade pip && \
    cd python && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    cd .. && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"


# path
ENV PYCAFFE_ROOT=$CAFFE_ROOT/python
ENV PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH
ENV PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$OCNN_ROOT/octree/build:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
