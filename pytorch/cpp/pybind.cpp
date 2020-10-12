#include "ocnn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("octree_batch", &octree_batch, "merge a batch of octrees");
  m.def("octree_samples", &octree_samples, "octree samples");
  m.def("points2octree", &points2octree, "convert points to octree");
  m.def("octree2col", &octree2col, "octree2col");
  m.def("col2octree", &col2octree, "col2octree");
  m.def("octree_conv", &octree_conv, "octree based convolution");
  m.def("octree_conv_grad", &octree_conv_grad, "octree based convolution");
  m.def("octree_pad", &octree_pad, "octree pad");
  m.def("octree_depad", &octree_depad, "octree depad");
  m.def("octree_max_pool", &octree_max_pool, "octree max pooling");
  m.def("octree_max_unpool", &octree_max_unpool, "octree max unpooling");
  m.def("octree_mask_pool", &octree_mask_pool, "octree mask pooling");
  m.def("octree_property", &octree_property, "get the octree property");
  m.def("transform_points", &transform_points, "transform the point cloud");
  m.def("normalize_points", &normalize_points, "normalize the point cloud");
  m.def("bounding_sphere", &bounding_sphere, "calc the bounding sphere");
}