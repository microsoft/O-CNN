#pragma once

#include <string>
#include <vector>
#include <torch/extension.h>

using std::string;
using std::vector;
using torch::Tensor;

vector<float> bounding_sphere(Tensor data_in, string method);
Tensor normalize_points(Tensor data_in, float radius, vector<float> center);
Tensor transform_points(Tensor data_in, vector<float> angle, vector<float> scale,
                        vector<float> jitter, float offset, string normal_axis);
vector<Tensor> clip_points(Tensor data_in, vector<float> bbmin, vector<float> bbmax);

Tensor octree_batch(vector<Tensor> tensors_in);
vector<Tensor> octree_samples(vector<string> names);
Tensor octree_property(Tensor octree_in, string property, int depth);
Tensor octree_set_property(Tensor octree_in, Tensor data_in, int depth);
Tensor points2octree(Tensor points, int depth, int full_depth, bool node_dis,
                     bool node_feature, bool split_label, bool adaptive,
                     int adp_depth, float th_normal, float th_distance,
                     bool extrapolate, bool save_pts, bool key2xyz);

Tensor octree2col(Tensor data_in, Tensor octree, int depth,
                  vector<int> kernel_size, int stride, bool nempty);
Tensor col2octree(Tensor grad_in, Tensor octree, int depth,
                  vector<int> kernel_size, int stride, bool nempty);

Tensor octree_conv(Tensor data_in, Tensor weights, Tensor octree, int depth,
                   int num_output, vector<int> kernel_size, int stride,
                   bool nempty);
Tensor octree_deconv(Tensor data_in, Tensor weights, Tensor octree, int depth,
                     int num_output, vector<int> kernel_size, int stride,
                     bool nempty);
vector<Tensor> octree_conv_grad(Tensor data_in, Tensor weights, Tensor octree,
                                Tensor grad_in, int depth, int num_output,
                                vector<int> kernel_size, int stride,
                                bool nempty);
vector<Tensor> octree_deconv_grad(Tensor data_in, Tensor weights, Tensor octree,
                                  Tensor grad_in, int depth, int num_output,
                                  vector<int> kernel_size, int stride,
                                  bool nempty);

Tensor octree_pad(Tensor data_in, Tensor octree, int depth, float val = 0.0f);
Tensor octree_depad(Tensor data_in, Tensor octree, int depth);

vector<Tensor> octree_max_pool(Tensor data_in, Tensor octree, int depth);
Tensor octree_max_unpool(Tensor data_in, Tensor mask, Tensor octree, int depth);
Tensor octree_mask_pool(Tensor data_in, Tensor mask, Tensor octree, int depth);

Tensor octree_encode_key(Tensor xyz);
Tensor octree_decode_key(Tensor key);
Tensor octree_key2xyz(Tensor key, int depth);
Tensor octree_xyz2key(Tensor xyz, int depth);
Tensor octree_search_key(Tensor key, Tensor octree, int depth, bool key_is_xyz,
                         bool nempty);

Tensor octree_scan(Tensor octree, vector<float> axis, float scale);
Tensor octree_grow(Tensor octree_in, int target_depth, bool full_octree);
Tensor octree_update(Tensor octree_in, Tensor label_in, int depth, int split);
Tensor octree_new(int batch_size, int channel, bool node_dis, int adaptive_layer);

vector<Tensor> octree_align(Tensor src_data, Tensor src_octree,
                            Tensor des_octree, int depth);
Tensor octree_align_grad(Tensor des_grad, Tensor idx_tensor);

Tensor points_new(Tensor pts, Tensor normals, Tensor features, Tensor labels);
Tensor points_property(Tensor points, string property);
Tensor points_batch_property(vector<Tensor> tensors_in, string property);
Tensor points_set_property(Tensor points, Tensor data, string property);
