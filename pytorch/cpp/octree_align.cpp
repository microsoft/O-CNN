#ifndef KEY64
#define KEY64
#endif
#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

vector<Tensor> octree_align(Tensor src_data, Tensor src_octree,
                            Tensor des_octree, int depth) {
  // in data
  src_data = src_data.contiguous();
  float* src_ptr = src_data.data_ptr<float>();
  int src_h = src_data.size(2);
  int channel = src_data.size(1);

  // octrees
  OctreeParser src_parser, des_parser;
  src_parser.set_gpu(src_octree.data_ptr<uint8_t>());
  des_parser.set_gpu(des_octree.data_ptr<uint8_t>());
  int des_h = des_parser.info().node_num(depth);
  CHECK_EQ(src_parser.info().node_num(depth), src_h);

  // get key
  torch::TensorOptions options = src_octree.options();
  const uintk* src_key = src_parser.key_gpu(depth);
  Tensor src_key_tensor;
  if (src_parser.info().is_key2xyz()) {
    src_key_tensor = torch::zeros({src_h}, options.dtype(torch::kInt64));
    uintk* ptr = (uintk*)src_key_tensor.data_ptr<int64_t>();
    xyz2key_gpu(ptr, src_key, src_h, depth);
    src_key = ptr;
  }

  const uintk* des_key = des_parser.key_gpu(depth);
  Tensor des_key_tensor;
  if (des_parser.info().is_key2xyz()) {
    des_key_tensor = torch::zeros({des_h}, options.dtype(torch::kInt64));
    uintk* ptr = (uintk*)des_key_tensor.data_ptr<int64_t>();
    xyz2key_gpu(ptr, des_key, des_h, depth);
    des_key = ptr;
  }

  // binary search
  Tensor idx_tensor = torch::zeros({src_h}, options.dtype(torch::kInt32));
  int* idx_ptr = idx_tensor.data_ptr<int>();
  search_key_gpu(idx_ptr, des_key, des_h, src_key, src_h);

  // out data
  Tensor des_tensor =
      torch::zeros({1, channel, des_h, 1}, options.dtype(torch::kFloat32));
  float* des_ptr = des_tensor.data_ptr<float>();

  // exec
  align_forward_gpu(des_ptr, des_h, channel, src_ptr, src_h, idx_ptr);
  return {des_tensor, idx_tensor};
}

Tensor octree_align_grad(Tensor des_grad, Tensor idx_tensor) {
  // gradients
  des_grad = des_grad.contiguous();
  float* des_ptr = des_grad.data_ptr<float>();
  int channel = des_grad.size(1);
  int des_h = des_grad.size(2);

  // index
  int src_h = idx_tensor.size(0);
  int* idx_ptr = idx_tensor.data_ptr<int>();

  // grad out
  torch::TensorOptions options = des_grad.options();
  Tensor src_tensor = torch::zeros({1, channel, src_h, 1}, options);
  float* src_ptr = src_tensor.data_ptr<float>();

  // exec
  align_backward_gpu(des_ptr, des_h, channel, src_ptr, src_h, idx_ptr);
  return src_tensor;
}
