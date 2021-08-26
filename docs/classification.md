# Shape Classification

Change the working directory to `caffe/experiments/cls` for `Caffe` and
`tensorflow/scripts` for `TensorFlow`.


## O-CNN on Caffe

1. Download [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) dataset
and unzip it to the folder `dataset/ModelNet40`.

2. Convert triangle meshes (in `off` format) to point clouds (in `points` format)
with the [virtual_scanner](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner).
This process can be automatically executed by the following command.
Remember to provide actual `<The path of the virtual_scanner>` to run the command,
and replace the symbol `^` with `\` for multiple-line commands in the shell.
We also provide the converted `point clouds` for convenience. Download the zip file
[here](https://www.dropbox.com/s/m233s9eza3acj2a/ModelNet40.points.zip?dl=0) and
unzip it to the folder `dataset/ModelNet40.points`.
    ```shell
    python prepare_dataset.py --run=m40_convert_mesh_to_points ^
                              --scanner=<The path of the virtual_scanner>
    ```

3. The generated point clouds are very dense, and if you would like to save disk
spaces, you can optionally run the following command to simplify the point cloud.
    ```shell
    python prepare_dataset.py --run=m40_simplify_points ^
                              --simplify_points=<The path of simplify_points>
    ```

4. Convert the point clouds to octrees, then build the `lmdb` database used by
`caffe` with the executive files [`octree`](Installation.md#Octree) and
[`convert_octree_data`](Installation.md#Caffe).
This process can be automatically executed by the following command.
Remember to provide actual `<The path of the octree>` and
`<The path of the convert_octree_data>`to run the command.
We also provide the converted `lmdb` for convenience. Download the zip file
[here](https://www.dropbox.com/s/t6d7z12ye3rpfit/ModelNet40.octree.lmdb.zip?dl=0)
and unzip it to the folder `dataset`.
    ```shell
    python prepare_dataset.py --run=m40_generate_ocnn_lmdb ^
                              --octree=<The path of the octree> ^
                              --converter=<The path of the convert_octree_data>
    ```

5. Run [`caffe`](Installation.md#Caffe) to train the model.
For detailed usage of [`caffe`](Installation.md#Caffe), please refer to the
official tutorial [Here](http://caffe.berkeleyvision.org/tutorial/interfaces.html).
We also provide our pre-trained Caffe model in `models/ocnn_M40_5.caffemodel`.
The classification accuracy on the testing dataset is 89.6% as reported in our paper.
If the voting or view pooling operations are performed, the accuracy will be 90.4%
    ```shell
    caffe train  --solver=ocnn_m40_5_solver.prototxt  --gpu=0
    ```


## AO-CNN on Caffe

1. Follow the instructions [above](#o-cnn-on-caffe) untile the 3rd step.
The AO-CNN takes adaptive octrees for input, run the following command to prepare
the data automatically.
    ```shell
    python prepare_dataset.py --run=m40_generate_aocnn_lmdb ^
                              --octree=<The path of the octree> ^
                              --converter=<The path of the convert_octree_data>
    ```
2. Run [`caffe`](Installation.md#Caffe) to train the model. The trained model
   and log can be downloaded
   [here](https://www.dropbox.com/s/r417fgq96wlzzj5/aocnn_m40_5.zip?dl=0).
    ```shell
    caffe train  --solver=aocnn_m40_5_solver.prototxt  --gpu=0
    ```


## O-CNN on TensorFlow

### Prepare the point cloud for ModelNet40   
1. Change the working directory to `tensorflow/data`. Run the following command
   to download [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip)
   dataset and unzip it to the folder `dataset/ModelNet40` via the following
   command:
    ```
    python cls_modelnet.py --run download_m40
    ```

2. Convert triangle meshes (in `off` format) to point clouds (in `points` format)
   with the [virtual_scanner](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner).
   This process can be automatically executed by the following command.
   Remember to provide actual `<The path of the virtual_scanner>` to run the command.
    ```shell
    python cls_modelnet.py --run m40_convert_mesh_to_points \
                           --scanner <The path of the virtual_scanner>
    ```

3. The generated point clouds are very dense, and if you would like to save disk
   spaces, you can optionally run the following command to simplify the point cloud.
    ```shell
    python cls_modelnet.py --run m40_simplify_points 
    ```
   We also provide the converted `point clouds` for convenience. Run the
   following command to download the zip file
   [here](https://www.dropbox.com/s/m233s9eza3acj2a/ModelNet40.points.zip?dl=0).
   ```shell
   python cls_modelnet.py --run download_m40_points
   ```

### Train a shallow O-CNN network
1. The `Tensorflow` takes `TFRecords` as input, run the following command to
   convert the point clouds to octrees, then build the `TFRecords` database
    ```shell
    python cls_modelnet.py --run m40_generate_octree_tfrecords 
    ```

2. Change the working directory to `tensorflow/script`. Run the following
   command to train the network. The network is based on the LeNet architecture.
   The performance is consistent with the `Caffe`-based implementation,  i.e.
   the classification accuracy is 89.6% without voting.
    ```shell
    python run_cls.py --config configs/cls_octree.yaml
    ```

### Train a deep O-CNN
1. With `Tensorflow`, the network can also directly consume the points as input
   and build octrees at runtime. Change the working directory to
   `tensorflow/data`. Run the following command to store the `points` into one
   `TFRecords` database.
    ```shell
    python cls_modelnet.py --run m40_generate_points_tfrecords
    ```
    
2. Change the working directory to `tensorflow/script`. Run the following command
   to train a **deeper** network with ResBlocks, which directly takes points.
   Notable, simply using the training hyperparameters as before, the testing
   accuracy increases from 89.6% to **92.4%**.
    ```shell
    python run_cls.py --config configs/cls_points.yaml
    ```

### Train a deep O-CNN-based HRNet
1. Change the working directory to `tensorflow/data`. Run the following command
   to store the `points` into `TFRecords` databases with different ratios of
   training data. Here we also rotate the upright axis of shapes from `z` axis
   to `y` axis.
    ```shell
    python cls_modelnet.py --run m40_generate_points_tfrecords_ratios
    ```
    
2. Change the working directory to `tensorflow/script`. Run the following command
   to train a **HRNet** with different ratios of training data.
    ```shell
    python run_cls_cmd.py
    ```
    <!-- https://www.dropbox.com/s/lmqv1n1yyja5z1j/m40_weights_and_logs.zip?dl=0 -->

## O-CNN on PyTorch

### Prepare the point cloud for ModelNet40   
1. Change the working directory to `pytorch/projects`. Run the following command
   to download [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip)
   dataset and unzip it to the folder `dataset/ModelNet40` via the following
   command:
    ```
    python tools/modelnet.py --run download_m40
    ```

2. Convert triangle meshes (in `off` format) to point clouds (in `points` format)
   with the [virtual_scanner](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner).
   This process can be automatically executed by the following command.
   Remember to provide actual `<The path of the virtual_scanner>` to run the command.
    ```shell
    python tools/modelnet.py --run m40_convert_mesh_to_points \
                             --scanner <The path of the virtual_scanner>
    ```

3. The generated point clouds are very dense, and if you would like to save disk
   spaces, you can optionally run the following command to simplify the point cloud.
    ```shell
    python tools/modelnet.py --run m40_simplify_points 
    ```
   We also provide the converted `point clouds` for convenience. Run the
   following command to download the zip file
   [here](https://www.dropbox.com/s/m233s9eza3acj2a/ModelNet40.points.zip?dl=0).
   ```shell
   python tools/modelnet.py --run download_m40_points
   ```

4. Generate the filelists. 
   ```shell
   python tools/modelnet.py --run generate_points_filelist
   ```

### Train a classification with Pytorch-based O-CNN


1. Run the following command to train the network.
   ```
   python classification.py --config configs/cls_m40.yaml
   ```

2. To train a deep ResNet, run the following command.
   ```
   python classification.py --config configs/cls_m40.yaml \
                            SOLVER.logdir logs/m40/resnet \
                            MODEL.name resnet
   ```
