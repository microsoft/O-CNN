# Shape Autoencoder

## Autoencoder on Caffe

The experiment in our Adaptive O-CNN is based on `Caffe`, and the 
instructions are on our working list.

1. Download points with normals from this [link](https://cloud.enpc.fr/s/j2ECcKleA1IKNzk), 
  and download the rendered views from this [link](https://cloud.enpc.fr/s/S6TCx1QJzviNHq0). 
  Unzip them after downloading.

2. Run the script `caffe/experiments/dataset.py` to generate the lmdbs: 
   `oct_test_lmdb`, `oct_train_aug_lmdb`, `oct_train_lmdb`, `img_test_lmdb` and 
   `img_train_lmdb`, of which `oct_train_aug_lmdb` and `oct_test_lmdb` are used 
   in the autoencoder task.

3. Download the protocol  buffer files `caffe/experiments/ae_7_4.train.prototxt` 
   and `caffe/experiments/ae_7_4.solver.prototxt`. 
   Configure the training and testing lmdb files and training the network.

4. After training, suppose the trained model is `autoencoder.caffemodel`, 
   download the protocol buffer file `caffe/experiments/ae_7_4.test.prototxt` 
   to test the network with the following command. 
   And the output octrees are dumped into the folder `ae_output`. 
   Then use the tools such as `octree2mesh`, `octree2points` and `points2ply` 
   to convert the octree into `obj` files or `ply` files, which can be visualized
   via some 3D viewers such as `MeshLab`. 

      caffe.exe test --model=ae_7_4.test.prototxt --weights=autoencoder.caffemodel ^
          --blob_prefix=ae_output/ae_test_ --gpu=0 --blob_header=false --iterations=[...]


## Autoencoder on TensorFlow

We implement an autoencoder on TensorFlow using the dataset for [shape completion](completion.md).
Following the instructions below to conduct the experiment.


1. Generate the datasets for training.  The data is originally provided by
   \[[Dai et al. 2017](http://graphics.stanford.edu/projects/cnncomple)\] and we
   convert the data to point clouds in the format of `ply`, which can be
   visualized via viewers like `meshlab`.  Run the following command to download
   the point clouds used for training and testing. 
    ```shell
    python data/completion.py --run generate_dataset
    ```

2. Change the working directory to `tensorflow/script`, and run the following
   command to train the autoencoder. The dimension of the hidden code is 2048.
    ```shell
    python run_ae.py --config configs/ae_resnet.yaml
    ```

3. To generate the shape in testing stage, run the following command.
    ```shell
    python run_ae.py --config configs/ae_resnet_decode.yaml
    ```