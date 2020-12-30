# Shape Autoencoder

## Autoencoder on Caffe

The experiment in our Adaptive O-CNN is based on `Caffe`. Before starting the
experiment please add the relavent executive files of `caffe` and `octree` to
the system path, since the following command will invoke them directly.

1. Download points with normals and rendered views from this
   [link](https://www.dropbox.com/s/3j4lnpmmplq61ue/aocnndata.zip?dl=0). Unzip
   them to the folder `caffe/dataset`.

2. Run the following command to generate the lmdbs:
   ```shell
   cd caffe/experiments
   python prepare_dataset.py --run=shapenet_lmdb_ae 
   ```

3. Run the following command to train the network:
   ```shell
   caffe train --solver=aocnn_ae_7_4.solver.prototxt --gpu=0
   ```

4. After training, run the following command to generate the results and
   calculate the chamder distances. The trained weights, log and chamder distances
   can be download from this [link](https://www.dropbox.com/s/nta8tnior85j6zn/aocnn_ae.zip?dl=0): 
   ``` shell
   mkdir dataset/ShapeNetV1.ae_output
   caffe test --model=aocnn_ae_7_4.test.prototxt \
              --weights=models/aocnn_ae_7_4_iter_350000.caffemodel \
              --blob_prefix=dataset/ShapeNetV1.ae_output/ae_test \
              --gpu=0 --blob_header=false --iterations=7943 
   python prepare_dataset.py --run=aocnn_ae_compute_chamfer
   ```


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
