# O-CNN: Octree-based Convolutional Neural Networks 

By [Peng-Shuai Wang](http://wang-ps.github.io/), [Yang Liu](https://xueyuhanlang.github.io/), Yu-Xiao Guo, Chun-Yu Sun and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/).

[Internet Graphics Group](https://www.microsoft.com/en-us/research/group/internet-graphics/), [Microsoft Research Asia](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/).

## Introduction 

This repository contains the implementation of O-CNN introduced in our Siggraph 2017 paper "[O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis](http://wang-ps.github.io/O-CNN.html)".  The code is released under the MIT license.

### Citation
If you use our code or models, please cite our paper.

    @article {Wang-2017-OCNN,
        title     = {O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis},
        author    = {Wang, Peng-Shuai and Liu, Yang and Guo, Yu-Xiao and Sun, Chun-Yu and Tong, Xin},
        journal   = {ACM Transactions on Graphics (SIGGRAPH)},
        volume    = {36},
        number    = {4},
        year      = {2017},
    }


## Installation

### O-CNN
O-CNN is built upon the [Caffe](https://github.com/BVLC/caffe) framework and it supports octree-based convolution, deconvolution, pooling, and unpooling. The code has been tested on the Windows and Linux platforms  (window 10 and Ubuntu 16.04), . Its installation is as follows:

- Clone [Caffe](https://github.com/BVLC/caffe) with revision `6bfc5ca8f7c2a4b7de09dfe7a01cf9d3470d22b3`
- The code for O-CNN is contained in the directory `caffe`. Clone and put it into the Caffe directory. 
- Follow the installation [instructions](https://github.com/BVLC/caffe/tree/windows) of Caffe to build the code to get the executive files `caffe.exe` and `convert_octree_data.exe` etc.

### Octree input for O-CNN
Our O-CNN takes the octree representation of 3D objects as input.  The efficient octree data structure is described in our paper. For convenience, we provide a reference implementation to convert the point cloud with oriented normal to our octree format. The code is contained in the directory `octree`, along with the Microsoft Visual studio 2015 solution file, which can be built to obtain the executable file `octree.exe`. 

## O-CNN in Action
The experiments in our paper can be reproduced as follows.

### Data preparation
For achieving better performance,  we store all the octree inputs in a  `leveldb` or `lmdb` database. Here are the details how to generate databases for O-CNN.

- Download and unzip the corresponding 3D model dataset (like the [ModelNet40](http://modelnet.cs.princeton.edu) dataset) into a folder. 

- Convert all the models (in OBJ/OFF format) to dense point clouds with normals (in POINTS format). As detailed in our paper, we build a virtual scanner and shoot rays to calculate the intersection points and oriented normals. The executable files and source code can be downloaded [here](https://github.com/wang-ps/O-CNN/tree/master/virtual%20scanner). 

- Run the tool `octree.exe` to convert point clouds into the octree files.

		Usage: Octree <filelist> [depth] [full_layer] [displacement] [augmentation] [segmentation]
			filelist: a text file whose each line specifies the full path name of a POINTS file
			depth: the maximum depth of the octree tree
			full_layer: which layer of the octree is full. suggested value: 2
			displacement: the offset value for handing extremely thin shapes: suggested value: 0.5
			segmentation: a boolean value indicating whether the output is for the segmentation task.


- Convert all the octrees into a `lmdb` or `leveldb` database by the tool `convert_octree_data.exe`.


### O-CNN for Shape Classification 
The instruction how to run the shape classification experiment:

- Download the [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) dataset, and convert it to a `lmdb` database as described above. [Here](https://www.dropbox.com/s/vzmxsqkp2lwwwp8/ModelNet40_5.zip?dl=0) we provide a `lmdb` database with 5-depth octrees for convenience.
- Download the `O-CNN` protocol buffer files, which are contained in the folder `caffe/examples/o-cnn`.
- Configure the path of the database and run `caffe.exe` according to the instructions of [Caffe](http://caffe.berkeleyvision.org/tutorial/interfaces.html). We also provide our pre-trained Caffe model in `caffe/examples/o-cnn`.

### O-CNN for Shape Retrieval
The instruction how to run the shape retrieval experiment:

- Download the dataset from  [SHREC16](http://shapenet.cs.stanford.edu/shrec16/), and convert it to a `lmdb` database as described above.
- Follow the same approach as the classification task to train the O-CNN.
- In the retrieval experiment, the orientation pooling is used to achieve better performance. The code `caffe/tools/feature_pooling.cpp` can be used to fulfill the task.   We will provide the automated tool soon.
- The retrieval result can be evaluated by the javascript code provided by [SHREC16](http://shapenet.cs.stanford.edu/shrec16/code/Evaluator.zip).

### O-CNN for Shape Segmentation
The instruction how to run the segmentation experiment: 

- The original part annotation data is provided as the supplemental material of the work "[A Scalable Active Framework for Region Annotation in 3D Shape Collections](http://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html)". As detailed in Section 5.3 of our paper, the point cloud in the original dataset is relatively sparse and the normal information is missing. We convert the sparse point clouds to dense points with normal information and correct part annotation.  Here is [one converted dataset](http://pan.baidu.com/s/1gfN5tPh) for your convenience, and the dense point clouds with segmentation labels can be downloaded [here](http://pan.baidu.com/s/1mieF2J2).
- Convert the dataset to a `lmdb` database.
- Download the protocol buffer files, which are contained in the folder `caffe/examples/o-cnn`. `NOTE:` as detailed in our paper, the training parameters are tuned and the pre-trained model from the retrieval task is used when the training dataset is relatively small. More details will be released soon.
- For CRF refinement, please refer to the code provided [here](https://github.com/wang-ps/O-CNN/tree/master/densecrf).  We will provide the automated tool soon.

## Acknowledgments

We thank the authors of [ModelNet](http://modelnet.cs.princeton.edu), [ShapeNet](http://shapenet.cs.stanford.edu/shrec16/) and [Region annotation dataset](http://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html) for sharing their 3D model datasets with the public.

## Contact

Please contact us (Pengshuai Wang wangps@hotmail.com, Yang Liu yangliu@microsoft.com ) if you have any problem about our implementation or request to access all the datasets.  

