# ScanNet Segmentation

### Data preparation

1. Download the data from the [ScanNet benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/).
   Unzip the data and place it to the folder `<scannet_folder>`

2. Change the working directory to `pytorch/projects`, run the following command
   to prepare the dataset.
   ```shell
   python tools/scannet.py --run process_scannet --path_in <scannet_folder>
   ```

3. Download the training, validation and testing file lists via the following command.
   ```
   python tools/scannet.py --run download_filelists
   ```


### Training and testing

1. Run the following command to train the network with 4 GPUs.
   ```shell
   python segmentation.py --config configs/seg_scannet.yaml SOLVER.gpu 0,1,2,3
   ```
   The mIoU on the validation set is 74.0, the training log and weights can be
   downloaded from this [link](https://www.dropbox.com/s/3grwj7vwd802yzz/D9_2cm.zip?dl=0). 

2. To achieve the 76.2 mIoU on the testing set, we follow the practice described
   in the MinkowsiNet, i.e. training the network both on the training and
   validation set via the following command.
   ```shell
   python segmentation.py --config configs/seg_scannet.yaml SOLVER.gpu 0,1,2,3  \
                          SOLVER.logdir logs/scannet/D9_2cm_all  \
                          DATA.train.filelist data/scannet/scannetv2_train_val_new.txt
   ```
   The training log and weights can be downloaded from this [link](https://www.dropbox.com/s/szhjus6kmknxyya/D9_2cm_all.zip?dl=0).

3. To generate the per-point predictions, run the following command.
   ```shell
   python segmentation.py --config configs/seg_scannet_eval.yaml
   ```

4. Run the following command to convert the predictions to segmentation labels.
   ```shell
   python tools/scannet.py --path_in  data/scannet/test  \
                           --path_pred logs/scannet/D9_2cm_eval  \
                           --path_out logs/scannet/D9_2cm_eval_seg   \
                           --filelist data/scannet/scannetv2_test_new.txt  \
                           --run generate_output_seg
   ```
   Then the generated segmentation results exist in the folder `logs/scannet/D9_2cm_eval_seg`