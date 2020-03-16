# O-CNN

<!-- ## Introduction <a name="introduction"></a> -->

This repository contains the implementation of *O-CNN*  and  *Aadptive O-CNN* 
introduced in our SIGGRAPH 2017 paper and SIGGRAPH Asia 2018 paper.  
The code is released under the **MIT license**.

We have released the TensorFlow-based implementation under the `tf` branch, and our future development will be focused on this implementation.
If you would like to have a try with the beta version, please pull the code and run the following command: `git checkout -b tf`.

- **[O-CNN: Octree-based Convolutional Neural Networks](https://wang-ps.github.io/O-CNN.html)**<br/>
  By [Peng-Shuai Wang](https://wang-ps.github.io/), [Yang Liu](https://xueyuhanlang.github.io/), 
  Yu-Xiao Guo, Chun-Yu Sun and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/) <br/>
  ACM Transactions on Graphics (SIGGRAPH), 36(4), 2017

- **[Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes](https://wang-ps.github.io/AO-CNN.html)**<br/>
By [Peng-Shuai Wang](https://wang-ps.github.io/), Chun-Yu Sun, [Yang Liu](https://xueyuhanlang.github.io/) 
and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/)<br/>
ACM Transactions on Graphics (SIGGRAPH Asia), 37(6), 2018<br/>


If you use our code or models, please cite our paper.
```
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
```

## Contents
### 1 &nbsp; [Installation](docs/Installation.md)
### 2 &nbsp; [Data preparation](docs/Data_Preparation.md)
### 3 &nbsp; [Shape Classification](docs/Classification.md)



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

