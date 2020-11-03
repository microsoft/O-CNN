# 3D Unsupervised Learning

## Unsupervised pretraining

1. Download the data and pretrained weights with the following command. 
   ```shell
   cd  tensorflow
   python data/midnet_data.py
   ```

2. Run the following command to train the network. 
   ```shell
   cd  script
   python run_mid.py --config configs/mid_hrnet_d6.yaml
   ```
   

## Finetune on PartNet

Follow the instructions
[here](docs/segmentation.md#shape-segmentation-on-partnet-with-tensorflow) to
download the PartNet and preprocess the data. 

Run the following script to finetune the network on PartNet with the pretrained
weights we provided. Compared with a random initialization, the IoU increases from
58.4 to 60.8. If you would like to finetune the network with your own pretrained
weights, you can simply provide the checkpoint via the command parameter `--ckpt`.

```shell
cd script
python run_seg_partnet_cmd.py --alias partnet_finetune --mode=finetune
```

In our paper, we also do experiments MidNet(Fix) in which the backbone network
is fixed and only the last two FC layers are trained. Run the following command
to reproduce the results.

```shell
cd script
python run_seg_partnet_cmd.py --alias partnet_fix --mode=fix
```

The trained weights and logs can also be downloaded (6.6G)
[here](https://www.dropbox.com/s/wrkcns19htdxb6x/partnet.zip?dl=0).
