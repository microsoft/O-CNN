# Shape Classification 

Change the working directory to `caffe/experiments/cls` for `Caffe` and 
`tensorflow/scripts` for `TensorFlow`.


## Data preparation
Download [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) dataset, 
convert it to `points`/`octree`.

Convert triangle meshes (in `off` format) to point clouds (in `points` format)
with the [virtualscanner](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner).
We provide the point clouds for convenience [Here](provide_the_link).

Convert the `points` to `octrees` with the following command:
```shell
octree --filenames filelist.txt --depth 5  --axis z
```

To generate the adaptive octree, run the following command:
```shell
octree --filenames filelist.txt --depth 5  --adaptive 1 --node_dis 1
```

## O-CNN on Caffe
Store the `octree` into one `lmdb` database.
[Here](https://www.dropbox.com/s/vzmxsqkp2lwwwp8/ModelNet40_5.zip?dl=0) we provide a 
`lmdb` database with 5-depth octrees for convenience, which can be directly unzipped.
```shell
convert_octree_data --shuffle <root_folder> <list_file> <lmdb_name>
```

The scripts for Caffe is contained in the folder `caffe/examples/cls`,
Configure the path of the database and run [`caffe`](Installation.md#Caffe) to 
train the model.
```shell
caffe train  --solver=ocnn_m40_5_solver.prototxt  --gpu=0
```
For detailed instructions, please refer to the official tutorial of 
[Caffe](http://caffe.berkeleyvision.org/tutorial/interfaces.html).
We also provide our pre-trained Caffe model in `caffe/examples/models`.


### AO-CNN on Caffe
Store the `octree` into one `lmdb` database.
<!-- [Here](https://www.dropbox.com/s/vzmxsqkp2lwwwp8/ModelNet40_5.zip?dl=0) we provide a 
`lmdb` database with 5-depth octrees for convenience, which can be directly unzipped. -->
```shell
convert_octree_data --shuffle <root_folder> <list_file> <lmdb_name>
```
The scripts for Caffe is contained in the folder `caffe/examples/cls`,
Configure the path of the database and run [`caffe`](Installation.md#Caffe) to 
train the model.
```shell
caffe train  --solver=aocnn_m40_5_solver.prototxt  --gpu=0
```


## O-CNN on TensorFlow
Store the `points` into one `TFRecord` database.
```shell
convert_tfrecords.py <root_folder> <list_file> <database>
```

Run the following command to train the network:
```
python cls
```
