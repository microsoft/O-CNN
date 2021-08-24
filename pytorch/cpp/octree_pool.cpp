#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

vector<Tensor> octree_max_pool(Tensor btm_data, Tensor octree, int depth) {
  // in octree
  OctreeParser octree_;
  octree_.set_gpu(octree.data_ptr<uint8_t>());

  // btm data
  btm_data = btm_data.contiguous();
  const float* btm_ptr = btm_data.data_ptr<float>();
  int channel = btm_data.size(1);
  int btm_h = btm_data.size(2);

  // check
  CHECK_EQ(octree_.info().node_num(depth), btm_h);
  CHECK_GT(depth, 1) << "Depth should be larger than 1";

  // top data
  int top_h = btm_h >> 3;
  torch::TensorOptions options = btm_data.options();
  Tensor top_data = torch::zeros({1, channel, top_h, 1}, options);
  float* top_ptr = top_data.data_ptr<float>();

  // mask
  Tensor mask = torch::zeros({1, channel, top_h, 1}, options.dtype(torch::kInt32));
  int* mask_ptr = mask.data_ptr<int>();

  // pooling
  octree_max_pool_gpu(top_ptr, top_h, mask_ptr, btm_ptr, btm_h, channel);
  return {top_data, mask};
}

Tensor octree_max_unpool(Tensor top_data, Tensor mask, Tensor octree, int depth) {
  // in octree
  OctreeParser octree_;
  octree_.set_gpu(octree.data_ptr<uint8_t>());

  // top data
  top_data = top_data.contiguous();
  const float* top_ptr = top_data.data_ptr<float>();
  int channel = top_data.size(1);
  int top_h = top_data.size(2);
  CHECK_EQ(top_h, octree_.info().node_num_nempty(depth - 1));

  // mask
  mask = mask.contiguous();
  const int* mask_ptr = mask.data_ptr<int>();
  CHECK(mask.size(1) == channel && mask.size(2) == top_h);

  // btm data
  int btm_h = top_h << 3;
  Tensor btm_data = torch::zeros({1, channel, btm_h, 1}, top_data.options());
  float* btm_ptr = btm_data.data_ptr<float>();

  // pooling
  octree_max_unpool_gpu(top_ptr, top_h, mask_ptr, btm_ptr, btm_h, channel);
  return btm_data;
}

Tensor octree_mask_pool(Tensor btm_data, Tensor mask, Tensor octree, int depth) {
  // in octree
  OctreeParser octree_;
  octree_.set_gpu(octree.data_ptr<uint8_t>());

  // btm data
  btm_data = btm_data.contiguous();
  const float* btm_ptr = btm_data.data_ptr<float>();
  int channel = btm_data.size(1);
  int btm_h = btm_data.size(2);

  // mask
  mask = mask.contiguous();
  auto mask_ptr = mask.data_ptr<int>();
  int top_h = mask.size(2);

  // check
  CHECK_EQ(octree_.info().node_num(depth), btm_h);
  CHECK_EQ(top_h, btm_h >> 3);

  // top data
  Tensor top_data = torch::zeros({1, channel, top_h, 1}, btm_data.options());
  float* top_ptr = top_data.data_ptr<float>();

  // pooling
  octree_mask_pool_gpu(top_ptr, top_h, mask_ptr, btm_ptr, btm_h, channel);
  return top_data;
}
