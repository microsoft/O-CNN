# Installation

## Octree

Our O-CNN takes the octree representation of 3D objects as input. 
And the octree can be built from a `point cloud` representation of a 3D shape.
The code for converting a point cloud into octree representation and some useful
tools to process octrees, point clouds and meshes are contained 
in the folder `octree`, which can be built via [cmake](https://cmake.org/):
```shell
cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git
cd .. && mkdir build && cd build
cmake ..  && cmake --build . --config Release
```


## Caffe

The caffe-based implementation has been tested on Windows 10 x64 and Ubuntu 16.04.
To build the code, [Cuda 8.0](https://developer.nvidia.com/cuda-downloads) and 
Cudnn 6.0 have to be installed.

- Clone [Caffe](https://github.com/BVLC/caffe) with revision `6bfc5ca`: 
```shell
git clone https://github.com/BVLC/caffe.git caffe-official
cd caffe-official && git checkout 6bfc5ca
```

- Copy the code contained in the directory `caffe` into the `caffe-official` directory to 
override the official [Caffe](https://github.com/BVLC/caffe) code. 

- Follow the installation [instructions](http://caffe.berkeleyvision.org/installation.html) 
of Caffe to build the code to get the executive files: `caffe`, `convert_octree_data` 
and `feature_pooling` etc.

- **NOTE**: Compared with the original code used in the experiments of the O-CNN paper, 
the code in this repository is refactored for the readability and maintainability, 
with the sacrifice of speed (it is about 10% slower, but it is more memory-efficient). 
If you want to try the original code or do some speed comparisons with our `O-CNN`,
feel free to drop me an email, we can share the original code with you. 


## Tensorflow

The code has been tested with Ubuntu 16.04/18.04 and TensorFlow 1.14.0/1.12.0.

- To build the code, [Cuda 10.1](https://developer.nvidia.com/cuda-downloads) and 
[Anaconda](https://www.anaconda.com/distribution/) with python 3.x have to be installed.

- Create a new conda environment and install tensorflow-gpu 1.14.0.
```shell
conda create -n tf-1.14.0 tensorflow-gpu==1.14.0
conda activate tf-1.14.0
```

- Build the code under `octree` with CUDA enabled.
```shell
cd octree/build
cmake .. -DUSE_CUDA=ON  && make
```

- Build the code under `tensorflow`.
```shell
cd tensorflow/libs
python build.py
```

- **NOTE**: If you install tensorflow via `pip` or `docker` instead of `conda`, 
you should install `g++4.8` and rebuild the code under folders `octree` and `tensorflow`.
If you get warning messages from numpy or BatchNorm, you can execute the following
commands: `pip install -U gast==0.2.2 numpy==1.16.4`. 

