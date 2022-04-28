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
<!-- cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..  && cmake --build . --config Release -->
After building, add the executive files to the system path using the following
command on the Win10.
```cmd
set PATH=the\absolute\path\octree\build\Release;%PATH%
```
Or run the following command on the Ubuntu.
```shell
export PATH=`pwd`:$PATH
```


## Caffe

The caffe-based implementation has been tested on Windows 10 x64 and Ubuntu 16.04.
To build the code, [Cuda 8.0](https://developer.nvidia.com/cuda-downloads) and 
Cudnn 6.0 have to be installed.

1. Clone [Caffe](https://github.com/BVLC/caffe) with revision `6bfc5ca`: 
    ```shell
    git clone https://github.com/BVLC/caffe.git caffe-official
    cd caffe-official && git checkout 6bfc5ca
    ```

2. Copy the code contained in the directory `caffe` into the `caffe-official` directory to 
override the official [Caffe](https://github.com/BVLC/caffe) code. 

3. Follow the installation [instructions](http://caffe.berkeleyvision.org/installation.html) 
of Caffe to build the code to get the executive files: `caffe`, `convert_octree_data` 
and `feature_pooling` etc.

4. **NOTE**: Compared with the original code used in the experiments of the O-CNN paper, 
the code in this repository is refactored for the readability and maintainability, 
with the sacrifice of speed (it is about 10% slower, but it is more memory-efficient). 
If you want to try the original code or do some speed comparisons with our `O-CNN`,
feel free to drop me an email, we can share the original code with you. 


<!-- 
cd caffe/docker
docker build --tag=ocnn:caffe gpu
docker run --runtime=nvidia --name=ocnn-caffe -it --rm ocnn:caffe /bin/bash 
docker pull wangps/ocnn:caffe
-->

## Tensorflow

The code has been tested with Ubuntu 16.04/18.04 and TensorFlow 1.14.0/1.12.0.

1. To build the code, [Cuda 10.1](https://developer.nvidia.com/cuda-downloads) and 
[Anaconda](https://www.anaconda.com/distribution/) with python 3.x have to be installed.

2. Create a new conda environment and install tensorflow-gpu 1.14.0.
    ```shell
    conda create -n tf-1.14.0 tensorflow-gpu==1.14.0
    conda activate tf-1.14.0
    conda install -c conda-forge yacs tqdm
    ```

3. Build the code under `tensorflow`.
    ```shell
    cd tensorflow/libs
    python build.py
    ```

5. **NOTE**: If you install tensorflow via `pip` or `docker` instead of `conda`, 
you should install `g++4.8` and rebuild the code under folders `octree` and `tensorflow`.
If you get warning messages from numpy or BatchNorm, you can execute the following
commands: `pip install -U gast==0.2.2 numpy==1.16.4`. 

## PyTorch

The code has been tested with Ubuntu 16.04 and PyTorch 1.6.0.

1. Enter the subfolder `pytorch`, and install PyTorch and relevant packages with
   the following commands:
    ```shell
    conda create --name pytorch-1.7.0 python=3.7
    conda activate pytorch-1.7.0
    conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
    pip install -r requirements.txt
    ```

    The code is also tested with the following pytorch version:
    - `pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1`
    - `pytorch==1.9.0 torchvision cudatoolkit=11.1 `
    <!--
    - `pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1` conda  failed
    - `pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel`        docker failed
    - `pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel`        docker succeed
    -->

2. Build O-CNN under PyTorch.
   ```shell
   python setup.py install --build_octree
   ```

3. Run the test cases.
   ```shell
   python -W ignore test/test_all.py -v
   ```