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
                        vector<float> jitter, float offset);

Tensor octree_batch(vector<Tensor> tensors_in);
vector<Tensor> octree_samples(vector<string> names);
Tensor octree_property(Tensor data_in, string property, int depth);
Tensor points2octree(Tensor points, int depth, int full_depth, bool node_dis,
                     bool node_feature, bool split_label, bool adaptive,
                     int adp_depth, float th_normal, float th_distance,
                     bool extrapolate, bool save_pts, bool key2xyz);

Tensor octree2col(Tensor data_in, Tensor octree, int depth,
                  vector<int> kernel_size, int stride);
Tensor col2octree(Tensor grad_in, Tensor octree, int depth,
                  vector<int> kernel_size, int stride);

Tensor octree_conv(Tensor data_in, Tensor weights, Tensor octree, int depth,
                   int num_output, vector<int> kernel_size, int stride);
vector<Tensor> octree_conv_grad(Tensor data_in, Tensor weights, Tensor octree,
                                Tensor grad_in, int depth, int num_output,
                                vector<int> kernel_size, int stride);

Tensor octree_pad(Tensor data_in, Tensor octree, int depth, float val = 0.0f);
Tensor octree_depad(Tensor data_in, Tensor octree, int depth);

vector<Tensor> octree_max_pool(Tensor data_in, Tensor octree, int depth);
Tensor octree_max_unpool(Tensor data_in, Tensor mask, Tensor octree, int depth);
Tensor octree_mask_pool(Tensor data_in, Tensor mask, Tensor octree, int depth);
