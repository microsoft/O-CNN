# O-CNN

<!-- ## Introduction <a name="introduction"></a> -->

This repository contains the implementation of our papers related with *O-CNN*.  
The code is released under the **MIT license**.

- **[O-CNN: Octree-based Convolutional Neural Networks](https://wang-ps.github.io/O-CNN.html)**<br/>
  By [Peng-Shuai Wang](https://wang-ps.github.io/), [Yang Liu](https://xueyuhanlang.github.io/), 
  Yu-Xiao Guo, Chun-Yu Sun and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/) <br/>
  ACM Transactions on Graphics (SIGGRAPH), 36(4), 2017

- **[Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes](https://wang-ps.github.io/AO-CNN.html)**<br/>
By [Peng-Shuai Wang](https://wang-ps.github.io/), Chun-Yu Sun, [Yang Liu](https://xueyuhanlang.github.io/) 
and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/)<br/>
ACM Transactions on Graphics (SIGGRAPH Asia), 37(6), 2018<br/>

- **[Deep Octree-based CNNs with Output-Guided Skip Connections for 3D Shape and Scene Completion](https://arxiv.org/abs/2006.03762)**<br/>
By [Peng-Shuai Wang](https://wang-ps.github.io/), [Yang Liu](https://xueyuhanlang.github.io/) 
and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/)<br/>
Computer Vision and Pattern Recognition (CVPR) Workshops, 2020<br/>

- **[Unsupervised 3D Learning for Shape Analysis via Multiresolution Instance Discrimination](https://arxiv.org/abs/2008.01068)**<br/>
By [Peng-Shuai Wang](https://wang-ps.github.io/), Yu-Qi Yang, Qian-Fang Zou, 
[Zhirong Wu](https://www.microsoft.com/en-us/research/people/wuzhiron/), 
[Yang Liu](https://xueyuhanlang.github.io/) 
and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/)<br/>
AAAI Conference on Artificial Intelligence (AAAI), 2021. [Arxiv, 2020.08]<br/>

If you use our code or models, please [cite](docs/citation.md) our paper.



### Contents
- [Installation](docs/installation.md)
- [Data Preparation](docs/data_preparation.md)
- [Shape Classification](docs/classification.md)
- [Shape Retrieval](docs/retrieval.md)
- [Shape Segmentation](docs/segmentation.md)
- [Shape Autoencoder](docs/autoencoder.md)
- [Shape Completion](docs/completion.md)
- [Image2Shape](docs/image2shape.md)
- [Unsupverised Pretraining](docs/unsupervised.md)
- [ScanNet Segmentation](docs/scannet.md)




### What's New?
- 2021.08.24: Update the code for pythorch-based O-CNN, including a UNet and
  some other major components. Our vanilla implementation without any tricks on
  [ScanNet](docs/scannet.md) dataset achieves 76.2 mIoU on the 
  [ScanNet benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/), even surpassing the
  recent state-of-art approaches published in CVPR 2021 and ICCV 2021.
- 2021.03.01: Update the code for pytorch-based O-CNN, including a ResNet and
  some important modules.
- 2021.02.08: Release the code for ShapeNet segmentation with HRNet.
- 2021.02.03: Release the code for ModelNet40 classification with HRNet.
- 2020.10.12: Release the initial version of our O-CNN under PyTorch. The code
  has been tested with the [classification task](docs/classification.md#o-cnn-on-pytorch).
- 2020.08.16: We released our code for [3D unsupervised learning](docs/unsupervised.md).
  We provided a unified network architecture for generic shape analysis tasks and 
  an unsupervised method to pretrain the network. Our method achieved state-of-the-art 
  performance on several benchmarks.
- 2020.08.12: We released our code for 
  [Partnet segmentation](docs/segmentation.md#shape-segmentation-on-partnet-with-tensorflow).
  We achieved  an average IoU of **58.4**, significantly better than PointNet
  (IoU: 35.6), PointNet++ (IoU: 42.5), SpiderCNN (IoU: 37.0), and PointCNN(IoU:
  46.5).
- 2020.08.05: We released our code for [shape completion](docs/completion.md).
  We proposed a simple yet efficient network and output-guided skip connections
  for 3D completion, which achieved state-of-the-art performances on several 
  benchmarks.


Please contact us (Peng-Shuai Wang wangps@hotmail.com, Yang Liu yangliu@microsoft.com ) 
if you have any problems about our implementation.  

