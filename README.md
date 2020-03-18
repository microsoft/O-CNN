# O-CNN

<!-- ## Introduction <a name="introduction"></a> -->

This repository contains the implementation of *O-CNN*  and  *Aadptive O-CNN* 
introduced in our SIGGRAPH 2017 paper and SIGGRAPH Asia 2018 paper.  
The code is released under the **MIT license**.

- **[O-CNN: Octree-based Convolutional Neural Networks](https://wang-ps.github.io/O-CNN.html)**<br/>
  By [Peng-Shuai Wang](https://wang-ps.github.io/), [Yang Liu](https://xueyuhanlang.github.io/), 
  Yu-Xiao Guo, Chun-Yu Sun and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/) <br/>
  ACM Transactions on Graphics (SIGGRAPH), 36(4), 2017

- **[Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes](https://wang-ps.github.io/AO-CNN.html)**<br/>
By [Peng-Shuai Wang](https://wang-ps.github.io/), Chun-Yu Sun, [Yang Liu](https://xueyuhanlang.github.io/) 
and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/)<br/>
ACM Transactions on Graphics (SIGGRAPH Asia), 37(6), 2018<br/>


If you use our code or models, please cite our paper.
```
@article {Wang-2017-OCNN,
    title     = {{O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis}},
    author    = {Wang, Peng-Shuai and Liu, Yang and Guo, Yu-Xiao and Sun, Chun-Yu and Tong, Xin},
    journal   = {ACM Transactions on Graphics (SIGGRAPH)},
    volume    = {36},
    number    = {4},
    year      = {2017},
}
@article {Wang-2018-AOCNN,
    title     = {{Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes}},
    author    = {Wang, Peng-Shuai and Sun, Chun-Yu and Liu, Yang and Tong, Xin},
    journal   = {ACM Transactions on Graphics (SIGGRAPH Asia)},
    volume    = {37},
    number    = {6},
    year      = {2018},
}
```

### Contents
- [Installation](docs/installation.md)
- [Data Preparation](docs/data_preparation.md)
- [Shape Classification](docs/classification.md)
- [Shape Retrieval](docs/retrieval.md)
- [Shape Segmentation](docs/segmentation.md)
- [Shape Autoencoder](docs/autoencoder.md)
- [Image2Shape](docs/image2shape.md)



We thank the authors of [ModelNet](http://modelnet.cs.princeton.edu), 
[ShapeNet](http://shapenet.cs.stanford.edu/shrec16/) and 
[Region annotation dataset](http://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html) 
for sharing their 3D model datasets with the public.

Please contact us (Pengshuai Wang wangps@hotmail.com, Yang Liu yangliu@microsoft.com ) 
if you have any problems about our implementation.  

