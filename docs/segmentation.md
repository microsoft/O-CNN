# Shape Segmentation

The experiment is based on `Caffe`. 
It is also possible to conduct this experiment with `TensorFlow`, and the 
instructions are on our working list.


- The original part annotation data is provided as the supplemental material of 
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

- Run the `octree.exe` to convert these point clouds to octree files. 
  Then you can get the octree files, which also contains the segmentation label.

- Convert the dataset to a `lmdb` database. 
  Since the segmentation label is contained in each octree file, the object label 
  for each octree file can be set to any desirable value. 
  And the object label is just ignored in the segmentation task.

- Download the protocol buffer files, which are contained in the folder `caffe/experiments`. 
  
- In the testing stage, the output label and probability of each finest leaf node 
  can also be obtained. Specifically, 
  open the file `segmentation_5.prototxt`, uncomment line 458~485, 
  set the `batch_size` in line 31  to 1, and run the following command to dump the result.  

      caffe.exe test --model=segmentation_5.prototxt --weights=segmentation_5.caffemodel --gpu=0
      --blob_prefix=feature/segmentation_5_test_ --binary_mode=false --save_seperately=true --iterations=[...]


- For CRF refinement, please refer to the code provided 
  [here](https://github.com/wang-ps/O-CNN/tree/master/densecrf).  