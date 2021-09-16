#include "ocnn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("octree_batch", &octree_batch, "merge a batch of octrees");
  m.def("octree_samples", &octree_samples, "octree samples");
  m.def("points2octree", &points2octree, "convert points to octree");
  m.def("octree2col", &octree2col, "octree2col");
  m.def("col2octree", &col2octree, "col2octree");
  m.def("octree_conv", &octree_conv, "octree based convolution");
  m.def("octree_deconv", &octree_deconv, "octree based deconvolution");
  m.def("octree_conv_grad", &octree_conv_grad, "octree based convolution");
  m.def("octree_deconv_grad", &octree_deconv_grad, "octree based deconvolution");
  m.def("octree_pad", &octree_pad, "octree pad");
  m.def("octree_depad", &octree_depad, "octree depad");
  m.def("octree_max_pool", &octree_max_pool, "octree max pooling");
  m.def("octree_max_unpool", &octree_max_unpool, "octree max unpooling");
  m.def("octree_mask_pool", &octree_mask_pool, "octree mask pooling");
  m.def("octree_property", &octree_property, "get the octree property",
        py::arg("octree"), py::arg("property"), py::arg("depth") = 0);
  m.def("octree_set_property", &octree_set_property, "set the octree property",
       py::arg("octree"), py::arg("data"), py::arg("depth"));
  m.def("transform_points", &transform_points, "transform the point cloud");
  m.def("clip_points", &clip_points, "clip the points with unit boundingbox");
  m.def("normalize_points", &normalize_points, "normalize the point cloud");
  m.def("bounding_sphere", &bounding_sphere, "calc the bounding sphere");
  m.def("octree_encode_key", &octree_encode_key, "encode xyz-id to octree key");
  m.def("octree_decode_key", &octree_decode_key, "decode octree key to xyz-id");
  m.def("octree_xyz2key", &octree_xyz2key, "convert key from xyz order");
  m.def("octree_key2xyz", &octree_key2xyz, "convert key to xyz order");
  m.def("octree_search_key", &octree_search_key, "search key from octree",
        py::arg("key"), py::arg("octree"), py::arg("depth"),
        py::arg("key_is_xyz") = true, py::arg("nempty") = false);
  m.def("octree_scan", &octree_scan, "octree scanning", py::arg("octree"),
        py::arg("axis"), py::arg("scale") = 1.0);        
  m.def("octree_grow", &octree_grow, "octree growing", py::arg("octree"),
        py::arg("target_depth"), py::arg("full_octree") = false);
  m.def("octree_update", &octree_update, "update octree via splitting labels",
        py::arg("octree"), py::arg("label"), py::arg("depth"), py::arg("split") = 1);
  m.def("octree_new", &octree_new, "create a new octree",
        py::arg("batch_size") = 1, py::arg("channel") = 4,
        py::arg("node_dis") = true, py::arg("adaptive_layer") = 0);
  m.def("octree_align", &octree_align, "align octree data", py::arg("src_data"),
        py::arg("src_octree"), py::arg("des_octree"), py::arg("depth"));
  m.def("octree_align_grad", &octree_align_grad, "backward of align-octree-data");
  m.def("points_batch_property", &points_batch_property, "get batch of points' property");
  m.def("points_property", &points_property, "get the points' property");
  m.def("points_set_property", &points_set_property, "set the points' property");
  m.def("points_new", &points_new, "create new points with given input");
}
