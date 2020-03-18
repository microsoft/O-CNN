# Shape Autoencoder

The experiment is based on `Caffe`. 
It is also possible to conduct this experiment with `TensorFlow`, and the 
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
   Configure the training and tesing lmdb files and training the network.

4. After training, suppose the trained model is `autoencoder.caffemodel`, 
   download the protocol buffer file `caffe/experiments/ae_7_4.test.prototxt` 
   to test the network with the following command. 
   And the output octrees are dumped into the folder `ae_output`. 
   Then use the tools such as `octree2mesh`, `octree2points` and `points2ply` 
   to convert the octree into `obj` files or `ply` files, which can be visualized
   via some 3D viewer such as `MeshLab`. 

      caffe.exe test --model=ae_7_4.test.prototxt --weights=autoencoder.caffemodel ^
          --blob_prefix=ae_output/ae_test_ --gpu=0 --blob_header=false --iterations=[...]

