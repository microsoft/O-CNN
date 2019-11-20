# O-CNN

## Introduction <a name="introduction"></a>

This repository contains the implementation of *O-CNN*  and  *Aadptive O-CNN* introduced in our SIGGRAPH 2017 paper and SIGGRAPH Asia 2018 paper.  The code is released under the MIT license.

We have released the TensorFlow-based implementation under the `tf` branch, and our future development will be focused on this implementation.
If you would like to have a try with the beta version, please pull the code and run the following command: `git checkout -b tf`.

* **[O-CNN: Octree-based Convolutional Neural Networks](https://wang-ps.github.io/O-CNN.html)**<br/>
By [Peng-Shuai Wang](https://wang-ps.github.io/), [Yang Liu](https://xueyuhanlang.github.io/), Yu-Xiao Guo, Chun-Yu Sun and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/)<br/>
ACM Transactions on Graphics (SIGGRAPH), 36(4), 2017

* **[Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes](https://wang-ps.github.io/AO-CNN.html)**<br/>
By [Peng-Shuai Wang](https://wang-ps.github.io/), Chun-Yu Sun, [Yang Liu](https://xueyuhanlang.github.io/) and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/)<br/>
ACM Transactions on Graphics (SIGGRAPH Asia), 37(6), 2018<br/>


If you use our code or models, please cite our paper.

    @article {Wang-2017-OCNN,
        title     = {{O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis}},
        author    = {Wang, Peng-Shuai and Liu, Yang and Guo, Yu-Xiao and Sun, Chun-Yu and Tong, Xin},
        journal   = {ACM Transactions on Graphics (SIGGRAPH)},
        volume    = {36},
        number    = {4},
        year      = {2017},
    }
    @article {Wang-2018-AOCNN,
        title     = {{Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes}},
        author    = {Wang, Peng-Shuai and Sun, Chun-Yu and Liu, Yang and Tong, Xin},
        journal   = {ACM Transactions on Graphics (SIGGRAPH Asia)},
        volume    = {37},
        number    = {6},
        year      = {2018},
    }


## 1 &nbsp; Installation <a name="installation"></a>

### 1.1 &nbsp; Manual Setup
O-CNN is built upon the [Caffe](https://github.com/BVLC/caffe) framework and it supports octree-based convolution, deconvolution, pooling, and unpooling. The code has been tested on the Windows 10 x64 (It can also be built on the Ubuntu 16.04). Its installation is as follows:

- Clone [Caffe](https://github.com/BVLC/caffe) into the caffe directory with revision `6bfc5ca`: `git clone https://github.com/BVLC/caffe.git && cd caffe && git checkout 6bfc5ca`.
- Clone the code for O-CNN, then copy the code contained in the directory `O-CNN/caffe` into the caffe directory to override the original [Caffe](https://github.com/BVLC/caffe) code. 
- Follow the installation [instructions](http://caffe.berkeleyvision.org/installation.html) of Caffe to build the code to get the executive files `caffe`, `convert_octree_data` and `feature_pooling` etc. Note that the code should be compiled with c++11 support, which is not enabled by default for some compilers. So please check your compiler settings and enable the compiler to support c++11.
- Our O-CNN takes the octree representation of 3D objects as input. The code for converting a point cloud into octree representation is contained in the folder `O-CNN/ocnn/octree`, which can be built via [cmake](https://cmake.org/): `cd O-CNN/ocnn/octree && mkdir build && cd build && cmake ..  && cmake --build . --config Release` .
- We also provide one tool to pre-process meshes from online dataset, which can be downloaded [here](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner).


After the building, you will get the executable files which is useful for conducting the experiments: 

- [`virtualscanner`](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner) - used to convert obj/off files to points files  
- [`octree`](#octree) - used to convert point files to octree files  <!-- - [`octree2ply`](#octree-2-ply) - used to convert octree files to ply files   -->
- [`convert_octree_data`](#convert-octree-data) - used to convert octree files to lmdb files  
- `caffe` - executable for training / evaluating models  
- `feature_pooling` - pools features and outputs them to an lmdb  

**NOTE**: Compared with the original code used in the experiments of the O-CNN paper, the code in this repository is refactored for the readability and maintainability, with the sacrifice of speed (it is about 10% slower, but it is more memory-efficient). If you want to try the original code or do some speed comparisons with our `O-CNN`, feel free to drop me an email, we can share the original code with you. <br/>
<!-- To build the octree, the bounding sphere of the object is needed to be computed. The initial version of our code is built upon the bound sphere library from this [link](https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html). However, we remove it from our code due to the licence issue. To reproduce the results in our paper, it is highly recommended to download the [bound sphere library](https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html).  -->


### 1.2 &nbsp; Docker Setup (For Ubuntu only)
A docker build file is provided to automatically build your environments so you don't have to worry about project dependencies. To get your environment up and running, execute the following:

```
cd docker
docker build --network=host --tag=ocnn .
docker run --runtime=nvidia --network=host --name=ocnn -it ocnn /bin/bash
```

You will now find yourself in a container environment where you can automatically prepare datasets and train/test an o-cnn. 

**Many thanks** to David Pisani (@[dapisani](https://github.com/dapisani)) for his contribution to the docker setup!


## 2 &nbsp; Data preparation <a name="data"></a>

### 2.1 &nbsp; General procedure
- Download and unzip the corresponding 3D model dataset (like the [ModelNet40](http://modelnet.cs.princeton.edu) dataset) into a folder.
- Convert all the models (in OBJ/OFF format) to dense point clouds with normals (in `POINTS` format) with the tool [`virtualscanner`](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner).  This tool is specially designed for the 3D model from ModelNet40 and ShapeNet. If your mesh is of good quality, there are many more efficient methods to sample points.

```
Usage:  
    VirtualScanner <file_name> [nviews] [flags] [normalize]
        file_name: the name of the file (*.obj; *.off) to be processed.
        nviews: the number of views for scanning. Default: 6
        flags: Indicate whether to output normal flipping flag. Default: 0
        normalize: Indicate whether to normalize input mesh. Default: 0        
Example:
    VirtualScanner input.obj 14             // process the file input.obj
    VirtualScanner D:\data\ 14              // process all the obj/off files under the folder D:\Data
```

- Run the tool `octree` to convert point clouds into the octree files.
```
Usage: 
    octree
        --filenames  <A file contains the absolute filenames of input POINTS each line>
        [--adaptive  <Build adaptive octree>=0]
        [--adp_depth  <The starting depth of adaptive octree>=4]
        [--axis  <The upright axis of the input model>=z]
        [--depth  <The maximum depth of the octree>=6]
        [--full_depth  <The full layer of the octree>=2]
        [--key2xyz  <Convert the key to xyz when serialization>=0]
        [--node_dis  <Output per-node displacement>=0]
        [--node_feature  <Compute per node feature>=0]
        [--offset  <The offset value for handing thin shapes>=0.55]
        [--output_path  <The output path>=.]
        [--rot_num  <Number of poses rotated along the upright axis>=12]
        [--split_label  <Compute per node splitting label>=0]
        [--th_distance  <The threshold for simplifying octree>=2.0f]
        [--th_normal  <The threshold for simplifying octree>=0.2f]
Example:
    // process all the points into octrees of depth 5
    octree --filenames filelist.txt --depth 5  --axis z
```
- Store all the octree files in a  `leveldb` or `lmdb` database by the tool `convert_octree_data`, which serves as the input of Caffe. For the ones who are new to the [Caffe](http://caffe.berkeleyvision.org/) framework and do not know how to use the generated database to train and test the neural network, the tutorial on [this page](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) is highly recommanded. 
```
Usage: 
    convert_octree_data <rootfolder> <listfile> <db_name>
        rootfolder: base folder where db will be output and listed files are relative to
        listfile: file which contains a list of each octree file to be added to the leveldb. 
                  Each entry should be the [octree_file] [category_number]
        db_name: Name of db to be outputted
Example:
    convert_octree_data D:/octrees/ D:/octrees/list.txt D:/octrees_lmdb
```
<!-- 
### 2.2 &nbsp; Automated Dataset Setup (For Ubuntu only)
For the dataset `ModelNet10` and `ModelNet40`, we provide some scripts to automatically prepare the datasets. The code is contained in the python folder. (We will update the prepare_dataset.py to support other datasets such as ShapeNet55.)
```
Usage: 
    prepare_dataset.py [-h] --datadir DATADIR --dataset {ModelNet10,ModelNet40}
        --points_converter_path POINTS_CONVERTER_PATH
        --octree_converter_path OCTREE_CONVERTER_PATH
        --lmdb_converter_path LMDB_CONVERTER_PATH
        [--starting_action {Retrieve,Extract,Clean,CreatePoints,CreateOctree,CreateLmdb,Finished}]
        [--depth DEPTH] [--full_layer FULL_LAYER]
        [--displacement DISPLACEMENT] [--augmentation AUGMENTATION]
        [--for_segmentation]

        datadir: directory where to download and prepare dataset
        dataset: dataset to prepare
        points_converter_path: path to obj/off to points converter
        octree_converter_path: path to points to octree converter
        lmdb_converter_path: path to octree to lmdb converter
        starting_action: if specified, starting action to perform from. Otherwise, continue from last completed action.
        depth: depth of octrees to create (default 6)
        full_layer: layer in which octree is full (default 2)
        displacement: offset value for thin shapes (default 0.55)
        augmentation: number of model poses converted to octrees (default 24)
        for_segmentation: flags whether dataset is for segmentation
```
 -->
## 3 &nbsp; O-CNN in action

### 3.1 &nbsp; O-CNN for Shape Classification 
The instruction to run the shape classification experiment:

- Download the [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) dataset, and convert it to a `lmdb` database as described above. [Here](https://www.dropbox.com/s/vzmxsqkp2lwwwp8/ModelNet40_5.zip?dl=0) we provide a `lmdb` database with 5-depth octrees for convenience. Since we upgraded the octree format in this version of code, please run the following command to upgrade the lmdb: `upgrade_octree_database.exe <input lmdb> <output lmdb>`. When manually generating the dataset, note that the upright direction of the 3D models in the `ModelNet40` is `z` axis, so the octree command is: `octree --filenames filelist.txt --depth 5  --axis z`
- Download the `O-CNN` protocol buffer files, which are contained in the folder `caffe/examples/o-cnn`.
- Configure the path of the database and run `caffe.exe` according to the instructions of [Caffe](http://caffe.berkeleyvision.org/tutorial/interfaces.html). We also provide our pre-trained Caffe model in `caffe/examples/o-cnn`.

### 3.2 &nbsp; O-CNN for Shape Retrieval
The instruction to run the shape retrieval experiment:

- Download the dataset from  [SHREC16](http://shapenet.cs.stanford.edu/shrec16/), and convert it to a `lmdb` database as described above.  Note that the upright direction of the 3D models in the `ShapeNet55` is `Y` axis, so the octree command is: `octree --filenames filelist.txt --depth 5  --axis y`.
[Here](http://pan.baidu.com/s/1mieF2J2) we provide the lmdb databases with 5-depth octrees for convenience, just download the files prefixed with `S55` and un-zip them. Since we upgraded the octree format in this version of code, please run the following command to upgrade the lmdb: `upgrade_octree_database.exe <input lmdb> <output lmdb>`.
- Follow the same approach as the classification task to train the O-CNN with the `O-CNN` protocal files `S55_5.prototxt` and `solver_S55_5.prototxt`, which are contained in the folder `caffe/examples/o-cnn`.
- In the retrieval experiment, the `orientation pooling` is used to achieve better performance, which can be perfromed following the steps below.
    - Generate feature for each object. For example, to generate the feature for the training data, open the file `S55_5.prototxt`, uncomment line 275~283, set the `source` in line 27 to the `training lmdb`, set the `batch_size` in line 28  to 1, and run the following command.
            
            caffe.exe test --model=S55_5.prototxt --weights=S55_5.caffemodel --blob_prefix=feature/S55_5_train_ ^
            --gpu=0 --save_seperately=false --iterations=[the training object number]

    Similarly, the feature for the validation data and testing data can also be generated. Then we can get three binary files, `S55_5_train_feature.dat, S55_5_val_feature.dat and S55_5_test_feature.dat`, containing the features of the training, validation and testing data respectively.
    - Pool the features of the same object. There are 12 features for each object since each object is rotated 12 times. We use max-pooling to merge these features.
            
            feature_pooling.exe --feature=feature/S55_5_train_feature.dat --number=12 ^
            --dbname=feature/S55_5_train_lmdb --data=[the data list file name]

    Then we can get the feature of training, validation and testing data after pooling, contained in the `lmdb` database `S55_5_train_lmdb`, `S55_5_val_lmdb` and `S55_5_test_lmdb`.
    - Fine tune the `FC` layers of O-CNN, i.e. using the `solver_S55_5_finetune.prototxt` to re-train the `FC` layers.
            
            caffe.exe train --solver=solver_S55_5_finetune.prototxt --weights=S55_5.caffemodel

    - Finally, dump the probabilities of each testing objects. Open the file `S55_5_finetune.prototxt`, uncomment the line 120 ~ 129, set the `batch_size` in line 27 to 1, change the `source` in line 26 to `feature/S55_5_test_lmdb`, and run the following command.
            
            caffe.exe test --model=S55_5_finetune.prototxt --weights=S55_5_finetune.caffemodel ^
            --blob_prefix=feature/S55_test_ --gpu=0 --save_seperately=false --iterations=[...]
            
- Use the matlab script `retrieval.m`, contained in the folder `caffe/examples/o-cnn`, to generate the final retrieval result. And evaluated it by the javascript code provided by [SHREC16](http://shapenet.cs.stanford.edu/shrec16/code/Evaluator.zip).

### 3.3 &nbsp; O-CNN for Shape Segmentation
The instruction to run the segmentation experiment: 

- The original part annotation data is provided as the supplemental material of the work "[A Scalable Active Framework for Region Annotation in 3D Shape Collections](http://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html)". As detailed in Section 5.3 of our paper, the point cloud in the original dataset is relatively sparse and the normal information is missing. We convert the sparse point clouds to dense points with normal information and correct part annotation.  Here is [one converted dataset](http://pan.baidu.com/s/1gfN5tPh) for your convenience. Since we upgraded the octree format in this version of code, please run the following command to upgrade the lmdb: `upgrade_octree_database.exe  --node_label <input lmdb> <output lmdb>`. 
And the dense point clouds with `segmentation labels` can be downloaded [here](http://pan.baidu.com/s/1mieF2J2). Again, before using them, upgrade with the following command: `upgrade_points.exe --filenames <the txt file containing the filenames> --has_label 1 `.
- Run the `octree.exe` to convert these point clouds to octree files. Note that you should set the parameter `Segmentation` to 1 when running the `octree.exe`. Then you can get the octree files, which also contains the segmentation label.
- Convert the dataset to a `lmdb` database. Since the segmentation label is contained in each octree file, the object label for each octree file can be set to any desirable value. And the object label is just ignored in the segmentation task.
- Download the protocol buffer files, which are contained in the folder `caffe/examples/o-cnn`. `NOTE:` as detailed in our paper, the training parameters are tuned and the pre-trained model from the retrieval task is used when the training dataset is relatively small. More details will be released soon.
- In the testing stage, the output label and probability of each finest leaf node can also be obtained. Specifically, open the file `segmentation_5.prototxt`, uncomment line 458~485, , set the `batch_size` in line 31  to 1, and run the following command to dump the result.  

        caffe.exe test --model=segmentation_5.prototxt --weights=segmentation_5.caffemodel --gpu=0
        --blob_prefix=feature/segmentation_5_test_ --binary_mode=false --save_seperately=true --iterations=[...]


- For CRF refinement, please refer to the code provided [here](https://github.com/wang-ps/O-CNN/tree/master/densecrf).  

## 4 &nbsp; AO-CNN in Action

### 4.1 &nbsp; AO-CNN for Classification
- Download the [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) dataset, and convert it to a `lmdb` database as described above. To generating the adaptive octree, the octree command changes to: `octree --filenames filelist.txt --depth 5  --adaptive 1 --node_dis 1`. 
<!-- `TODO`: this parameter settings for `octree` is slightly different with the settings in our paper.  -->
- Download the protocol buffer files `cls_5.solver.prototxt` and `cls_5.prototxt`, which are contained in the folder `caffe/examples/ao-cnn`.
- Configure the path of the database and run `caffe.exe` according to the instructions of [Caffe](http://caffe.berkeleyvision.org/tutorial/interfaces.html). 


### 4.2 &nbsp; AO-CNN for Autoencoder

- Download points with normals from this [link](https://cloud.enpc.fr/s/j2ECcKleA1IKNzk), and download the rendered views from this [link](https://cloud.enpc.fr/s/S6TCx1QJzviNHq0). Unzip them after downloading.
- Run the script `O-CNN/caffe/examples/ao-cnn/dataset.py` to generate the lmdbs: `oct_test_lmdb`, `oct_train_aug_lmdb`, `oct_train_lmdb`, `img_test_lmdb` and `img_train_lmdb`, of which `oct_train_aug_lmdb` and `oct_test_lmdb` are used in the autoencoder task.
- Download the protocol  buffer files `O-CNN/caffe/examples/ao-cnn/ae_7_4.train.prototxt` and `O-CNN/caffe/examples/ao-cnn/ae_7_4.solver.prototxt`. Configure the training and tesing lmdb files and training the network.
- After training, suppose the trained model is `autoencoder.caffemodel`, download the protocol buffer file `O-CNN/caffe/examples/ao-cnn/ae_7_4.test.prototxt` to test the network with the following command. And the output octrees are dumped into the folder `ae_output`. Then use the tools such as `octree2mesh`, `octree2points` and `points2ply` to convert the octree into `obj` files or `ply` files, which can be visualized via some 3D viewer such as `MeshLab`. 

        caffe.exe test --model=ae_7_4.test.prototxt --weights=autoencoder.caffemodel ^
            --blob_prefix=ae_output/ae_test_ --gpu=0 --blob_header=false --iterations=[...]


<!-- ### 4.3 &nbsp; AO-CNN for Shape completion -->

### 4.4 &nbsp; AO-CNN for Image2Shape
- The lmdbs used in this experiment are `oct_test_lmdb`, `oct_train_lmdb`, `img_test_lmdb` and `img_train_lmdb`,  which are generated by the script `O-CNN/caffe/examples/ao-cnn/dataset.py`, as detailed in section 4.2.
- Download the protocol  buffer files `O-CNN/caffe/examples/ao-cnn/image2shape.train.prototxt` and `O-CNN/caffe/examples/ao-cnn/image2shape.solver.prototxt`. Configure the training and tesing lmdb files and training the network.
- After training, download the protocol buffer file `O-CNN/caffe/examples/ao-cnn/image2shape.test.prototxt` to testing the network. Follow the same procedure to dump and visualize the output as detailed in Section 4.2.



## 5 &nbsp; Acknowledgments

We thank the authors of [ModelNet](http://modelnet.cs.princeton.edu), [ShapeNet](http://shapenet.cs.stanford.edu/shrec16/) and [Region annotation dataset](http://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html) for sharing their 3D model datasets with the public.

## 6 &nbsp; Contact

Please contact us (Pengshuai Wang wangps@hotmail.com, Yang Liu yangliu@microsoft.com ) if you have any problem about our implementation or request to access all the datasets.  

