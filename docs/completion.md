# Shape Completion

We present a simple yet effective deep learning approach for completing the
input noisy and incomplete shapes or scenes. Our network is built upon the
octree-based CNNs (O-CNN)  with U-Net like structures and novel output-guided
skip-connections.

Following the instructions below to conduct the shape completion experiment.


1. Generate the datasets for training.  The data is originally provided by
   \[[Dai et al. 2017](http://graphics.stanford.edu/projects/cnncomple)\] and we
   convert the data to point clouds in the format of `ply`, which can be
   visualized via viewers like `meshlab`.  Run the following command to download
   the point clouds used for training and testing. 
    ```shell
    python data/completion.py --run generate_dataset
    ```

2. Train the network. Change the working directory via `cd ./script` and run the
   following command to train the network.
    ```shell
    python run_completion.py --config configs/completion_train.yaml 
    ```
   To train a pure autoencoder without the proposed output guided skip
   connections, run the following command.
    ```shell
    python run_completion.py --config configs/completion_train.yaml          \
           MODEL.skip_connections False SOLVER.logdir logs/completion/ae
    ```
   
   
3. Test the network. The testing dataset contains 12k partial scans, run the
   following command to test the trained model, and the output shapes in the
   format of `octree` are contained in the folder `logs/completion/skip_connections_test`.
   ```shell
   python run_completion.py --config configs/completion_test.yaml
   ```
   To test the autoencoder, run the following command, and the output shapes in the
   format of `octree` are contained in the folder `logs/completion/ae_test`.
   ```shell
   python run_completion.py --config configs/completion_test.yaml            \
          MODEL.skip_connections False SOLVER.logdir logs/completion/ae_test \
          SOLVER.ckpt logs/completion/ae/model/iter_320000.ckpt
   ```
   We also provide the pre-trained models, run the following command to have a
   quick test.
   ```shell
   python run_completion.py --config configs/completion_test.yaml            \
          SOLVER.ckpt dataset/ocnn_completion/models/skip_connections/iter_320000.ckpt
   ```

4. Get the completion results. Change the working directory via `cd ..`. Run the
   following command to convert the octree in the folder
   `logs/completion/skip_connections_test` to point cloud in the format of
   points/ply in the folder `script/dataset/ocnn_completion/output.points` and
   `script/dataset/ocnn_completion/output.ply`.
   ```shell
   python data/completion.py --run rename_output_octree
   python data/completion.py --run convert_octree_to_points 
   ```