# 3D Unsupervised Learning

## Unsupervised pretraining

1. Download the `ShapeNetCore.v1.zip` from [here](https://shapenet.org/) and
   place it in the folder `tensorflow/script/dataset/midnet_data`. Then run the
   following command to preprocess the data and make the tfrecords:
   ```shell
   cd  tensorflow/data
   python midnet_data.py --run shapenet_create_tfrecords \
                         --scanner <The path of the virtual_scanner>
   ```
2. For convenience, we also provide the tfrecords we made. Download the data and
   pretrained weights with the following command. 
   ```shell
   python midnet_data.py --run download_data
   ```

3. Run the following command to train the network. 
   ```shell
   cd  tensorflow/script
   python run_mid.py --config configs/mid_hrnet_d6.yaml
   ```
   
## Finetune on ModelNet40

Follow the instructions [here](classification.md#prepare-the-point-cloud-for-modelnet40) 
to download the ModelNet40 and preprocess the data. Make sure you can train the
HRNet with random initialization [here](classification.md#train-a-deep-o-cnn-based-hrnet).  

Then run the following script to finetune the network with the pretrained
weights we provided.  If you would like to finetune the network with your own
pretrained weights, you can simply provide the checkpoint via the command
parameter `--ckpt`.

```shell
python run_cls_cmd.py --alias m40_finetune --mode finetune
```

In our paper, we also do experiments MidNet(Fix) in which the backbone network
is fixed and only one linear classifier is trained. Run the following command.

```shell
python run_linear_cls_cmd.py --alias m40_linear
```


## Finetune on ShapeNetPart

Follow the instructions
[here](segmentation.md#shape-segmentation-on-shapeNet-with-tensorflow) to train
the HRNet with random initialization.  

Then run the following script to finetune the network with the pretrained
weights we provided.  If you would like to finetune the network with your own
pretrained weights, you can simply provide the checkpoint via the command
parameter `--ckpt`.

```shell
python run_seg_shapenet_cmd.py --alias shapenet_finetune --mode finetune
```

In our paper, we also do experiments MidNet(Fix) in which the backbone network
is fixed and only 2 FC layers are trained. Run the following command.

```shell
python run_seg_shapenet_cmd.py --alias shapenet_2fc --mode 2fc
```



## Finetune on PartNet

Follow the instructions
[here](segmentation.md#shape-segmentation-on-partnet-with-tensorflow) to
download the PartNet and preprocess the data.  

Change the working directory to `tensorflow/script`. Run the following script to
finetune the network on PartNet with the pretrained weights we provided.
Compared with a random initialization, the IoU increases from 58.4 to 60.8. If
you would like to finetune the network with your own pretrained weights, you can
simply provide the checkpoint via the command parameter `--ckpt`.

```shell
python run_seg_partnet_cmd.py --alias partnet_finetune --mode finetune
```

In our paper, we also do experiments MidNet(Fix) in which the backbone network
is fixed and only the last two FC layers are trained. Run the following command
to reproduce the results.

```shell
python run_seg_partnet_cmd.py --alias partnet_fix --mode fix
```

The trained weights and logs can also be downloaded (6.6G)
[here](https://www.dropbox.com/s/wrkcns19htdxb6x/partnet.zip?dl=0).



