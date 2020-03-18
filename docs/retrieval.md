
# Shape Retrieval

The shape retrieval experiment is based on `Caffe`. 
It is also possible to conduct this experiment with `TensorFlow`, and the 
instructions are on our working list.

1. Download the dataset from  [SHREC16](http://shapenet.cs.stanford.edu/shrec16/), 
   and convert it to a `lmdb` database as described in the 
   [classification experiment](docs/classification.md). 
   Note that the upright direction of the 3D models in the `ShapeNet55` is `Y` axis, 
   so the octree command is: `octree --filenames filelist.txt --depth 5  --axis y`.
   [Here](http://pan.baidu.com/s/1mieF2J2) we provide the lmdb databases with 5-depth 
   octrees for convenience, just download the files prefixed with `S55` and un-zip them. 
   Since we upgraded the octree format in this version of code, please run the following 
   command to upgrade the lmdb: `upgrade_octree_database.exe <input lmdb> <output lmdb>`.

2. Follow the same approach as the classification task to train the O-CNN with the 
   `O-CNN` protocal files `s55_5.prototxt` and `solver_s55_5.prototxt`, which are 
    contained in the folder `caffe/experiments`.

3. In the retrieval experiment, the `orientation pooling` is used to achieve better 
   performance, which can be performed following the steps below.
    - Generate feature for each object. For example, to generate the feature for 
      the training data, open the file `S55_5.prototxt`, uncomment line 275~283, 
      set the `source` in line 27 to the `training lmdb`, set the `batch_size` 
      in line 28  to 1, and run the following command.
            
            caffe.exe test --model=S55_5.prototxt --weights=S55_5.caffemodel --blob_prefix=feature/S55_5_train_ ^
            --gpu=0 --save_seperately=false --iterations=[the training object number]

      Similarly, the feature for the validation data and testing data can also be 
      generated. Then we can get three binary files, 
      `s55_5_train_feature.dat, s55_5_val_feature.dat and s55_5_test_feature.dat`, 
      containing the features of the training, validation and testing data respectively.
    - Pool the features of the same object. There are 12 features for each object 
      since each object is rotated 12 times. We use max-pooling to merge these features.
            
            feature_pooling.exe --feature=feature/S55_5_train_feature.dat --number=12 ^
            --dbname=feature/S55_5_train_lmdb --data=[the data list file name]

      Then we can get the feature of training, validation and testing data after pooling, 
      contained in the `lmdb` database `S55_5_train_lmdb`, `S55_5_val_lmdb` and `S55_5_test_lmdb`.

    - Fine tune the `FC` layers of O-CNN, i.e. using the `solver_s55_5_finetune.prototxt`
      to re-train the `FC` layers.
            
            caffe.exe train --solver=solver_S55_5_finetune.prototxt --weights=S55_5.caffemodel

    - Finally, dump the probabilities of each testing objects. Open the file
      `S55_5_finetune.prototxt`, uncomment the line 120 ~ 129, set the `batch_size` 
      in line 27 to 1, change the `source` in line 26 to `feature/S55_5_test_lmdb`,
      and run the following command.
            
            caffe.exe test --model=S55_5_finetune.prototxt --weights=S55_5_finetune.caffemodel ^
            --blob_prefix=feature/S55_test_ --gpu=0 --save_seperately=false --iterations=[...]
            
4. Use the matlab script `retrieval.m`, to generate the final retrieval result. 
   And evaluated it by the javascript code provided by 
   [SHREC16](http://shapenet.cs.stanford.edu/shrec16/code/Evaluator.zip).
