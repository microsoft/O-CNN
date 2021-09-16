# Shape Segmentation

## Shape Segmentation on ShapeNet with Caffe

The experiment is based on `Caffe`. 
It is also possible to conduct this experiment with `TensorFlow`, and the 
instructions are on our working list.


1. The original part annotation data is provided as the supplemental material of 
  this [paper](http://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html)". 
  As detailed in Section 5.3 of our paper, the point cloud in the original dataset 
  is relatively sparse and the normal information is missing. 
  We convert the sparse point clouds to dense points with normal information and 
  correct part annotation.  
  Here is [one converted dataset](http://pan.baidu.com/s/1gfN5tPh) for your convenience. 
  Since we upgraded the octree format in this version of code, please run the 
  following command to upgrade the lmdb: 
  `upgrade_octree_database.exe  --node_label <input lmdb> <output lmdb>`. 
  And the dense point clouds with `segmentation labels` can be downloaded 
  [here](http://pan.baidu.com/s/1mieF2J2). 
  Again, before using them, upgrade with the following command: 
  `upgrade_points.exe --filenames <the txt file containing the filenames> --has_label 1 `.

2. Run the `octree.exe` to convert these point clouds to octree files. 
  Then you can get the octree files, which also contains the segmentation label.

3. Convert the dataset to a `lmdb` database. 
  Since the segmentation label is contained in each octree file, the object label 
  for each octree file can be set to any desirable value. 
  And the object label is just ignored in the segmentation task.

4. Download the protocol buffer files, which are contained in the folder `caffe/experiments`. 
  
5. In the testing stage, the output label and probability of each finest leaf node 
  can also be obtained. Specifically, 
  open the file `segmentation_5.prototxt`, uncomment line 458~485, 
  set the `batch_size` in line 31  to 1, and run the following command to dump the result.  

      caffe.exe test --model=segmentation_5.prototxt --weights=segmentation_5.caffemodel --gpu=0
      --blob_prefix=feature/segmentation_5_test_ --binary_mode=false --save_seperately=true --iterations=[...]


6. For CRF refinement, please refer to the code provided 
  [here](https://github.com/wang-ps/O-CNN/tree/master/densecrf).  


## Shape Segmentation on PartNet with Tensorflow

We implement the [HRNet](https://github.com/HRNet) within our O-CNN framework,
and conduct the shape segmentation on
[PartNet](https://github.com/daerduoCarey/partnet_dataset). On the testing set
of PartNet, we achieve an average IoU of **58.4**, significantly better than
PointNet (IoU: 35.6), PointNet++ (IoU: 42.5), SpiderCNN (IoU: 37.0), and
PointCNN(IoU: 46.5). For more details, please refer to Section 4.3 of our paper
on [3D Unsupervised Learning](https://arxiv.org/abs/2008.01068).


1. Download [PartNet](https://github.com/daerduoCarey/partnet_dataset) according
   the provided instructions (about 100G). Then unzip the data to the folder 
   `tensorflow/script/dataset/partnet_segmentation`.
   
2. Run the following script to convert the original data to `tfrecords`. This
   process may take several hours to complete. For convenience, we also provide
   the converted `tfrecords` (6.6G)
   [here](https://www.dropbox.com/s/wrkcns19htdxb6x/partnet.zip?dl=0). Download
   and unzip the data to the folder `tensorflow/script/dataset`.
    ```shell
    cd tensorflow
    python data/seg_partnet.py 
    ```

3. Run the following script to train the network. The trained weights and
   logs can also be downloaded (6.6G)
   [here](https://www.dropbox.com/s/wrkcns19htdxb6x/partnet.zip?dl=0).
    ```shell
    cd script
    python run_seg_partnet_cmd.py
    ```


## Shape Segmentation on ShapeNet with Tensorflow

Follow the instructions below to train the HRNet based on our O-CNN framework 
and conduct the shape segmentation on ShapeNet Part. 

1. Change the working directory to `tensorflow/data`. Run the following script
   to download the data and convert the data to `tfrecords`. 
    ```shell
    python seg_shapenet.py 
    ```

3. Change the working directory to `tensorflow/script`. Run the following script
   to train the network. 
    ```shell
    python run_seg_shapenet_cmd.py
    ```


## Shape Segmentation on ShapeNet with Pytorch

Follow the instructions below to train a shallow SegNet used in our O-CNN paper,
which consists of pooling and unpooling layers.

1. Change the working directory to `tensorflow/data`. Run the following script
   to download the data, then make a symbolic link to the folder
   `pytorch/projects/dataset`:
    ```shell
    python seg_shapenet.py 
    ln -s `pwd`/../script/dataset/shapenet_segmentation `pwd`/../../pytorch/projects/data
    ```

3. Change the working directory to `pytorch/projects`. Run the following script
   to train the network. 
    ```shell
    python segmentation.py --config configs/seg_shapenet.yaml
    ```