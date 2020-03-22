# Data preparation

## General procedure

O-CNN takes `octrees` as input, which are built from `point clouds`.
We provide several tools for converting triangle meshes (in obj/off/ply format) 
into point clouds (in our customized `points` format), and converting
point clouds into octrees (in our customized `octree` format). 

- [`virtualscanner`](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner):
shoot parallel rays towards the object, then calculate the intersections of the rays 
and the surface, and orient the normals of the surface points towards the rays.
We used this tool in the experiments of our paper O-CNN and Adaptive O-CNN.

- [`mesh2points`](installation.md#Octree): uniformly sample points from the input 
object, and calculate the point normals via cross product.
`mesh2points` runs much faster than `virtualscanner`, but the point normals are 
not oriented. Use this tool if the mesh contains no flipped triangles.

- [`octree`](installation.md#Octree): convert point clouds into the octrees.


For better I/O performance, it is a good practice to store the `points`/`octree`
files into a database (`leveldb`/`lmdb` database for `caffe`, and `TFRecord` for
`tensorflow`).
We also provide tools for converting `points`/`octree` file into a database, or
or reverting the database.


- [`convert_octree_data`](installation.md#Caffe): used by `Caffe` to store `octree`
files to a `leveldb`/`lmdb` database.

- [`revert_octree_data`](installation.md#Caffe): used to revert a `leveldb`/`lmdb`
database to `octree` files.

- [`convert_tfrecords.py`](../tensorflow/util/convert_tfrecords.py):
used to `TensorFlow` to store `points`/`octree` files into a `TFRecord` database.

- [`revert_tfrecords.py`](../tensorflow/util/revert_tfrecords.py):
used to revert a `TFRecord` database to `points`/`octree` files.


## Custom data
It is also very convenient to write code to save your data into our `points` format.
Just include the header [points.h](../octree/octree/points.h) 
and refer to  the following several lines of code. 
An example can be found at [custom_data.cpp](../octree/tools/custom_data.cpp) 

```cpp
#include <points.h>

Points point_cloud;
vector<float> points, normals, features, labels;
// ......
// Set your data in points, normals, features, and labels.
// The points must not be empty, the labels may be empty,
// the normals & features must not be empty at the same time.
//   points: 3 channels, x_1, y_1, z_1, ..., x_n, y_n, z_n
//   normals: 3 channels, nx_1, ny_1, nz_1, ..., nx_n, ny_n, nz_n
//   features (such as RGB color): k channels, r_1, g_1, b_1, ..., r_n, g_n, b_n
//   labels: 1 channels, per-points labels
// ...... 
point_cloud.set_points(points, normals, features, labels);
point_cloud.write_points("my_points.points");
```

Moreover, you can also save your point cloud into a `PLY` points, and use the tool
[ply2points](installation.md#Octree) to convert the file to `points`.