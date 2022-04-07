#ifndef KEY64
#define KEY64
#endif
#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

// if KEY64 is defined, uintk is uint64

Tensor octree_encode_key(Tensor xyz) {
  xyz = xyz.contiguous();  // !!! make sure the Tensor is contiguous
  auto ptr_in = xyz.data_ptr<int16_t>();
  int num = xyz.size(0);
  int channel = xyz.size(1);
  CHECK_EQ(xyz.dim(), 2) << "The dim of input tensor must be 2.";
  CHECK_EQ(channel, 4) << "The channel of input tensor must be 4.";

  torch::TensorOptions options = xyz.options();
  Tensor data_out = torch::zeros({num}, options.dtype(torch::kInt64));
  auto ptr_out = data_out.data_ptr<int64_t>();
  if (xyz.is_cuda()) {
    memcpy_gpu(num, (uintk*)ptr_in, (uintk*)ptr_out);
  } else {
    memcpy_cpu(num, (uintk*)ptr_in, (uintk*)ptr_out);
  }
  return data_out;
}

Tensor octree_decode_key(Tensor key) {
  key = key.contiguous();  // !!! make sure the Tensor is contiguous
  auto ptr_in = key.data_ptr<int64_t>();
  int num = key.size(0);
  CHECK_EQ(key.dim(), 1) << "The dim of input tensor must be 1.";

  torch::TensorOptions options = key.options();
  Tensor data_out = torch::zeros({num, 4}, options.dtype(torch::kInt16));
  auto ptr_out = data_out.data_ptr<int16_t>();
  if (key.is_cuda()) {
    memcpy_gpu(num, (uintk*)ptr_in, (uintk*)ptr_out);
  } else {
    memcpy_cpu(num, (uintk*)ptr_in, (uintk*)ptr_out);
  }
  return data_out;
}

Tensor octree_xyz2key(Tensor xyz, int depth) {
  xyz = xyz.contiguous();  // !!! make sure the Tensor is contiguous
  auto ptr_in = xyz.data_ptr<int64_t>();
  int num = xyz.numel();
  CHECK_GE(num, 1) << "The numel of the input tensor must be larger than 1.";

  Tensor key = torch::zeros_like(xyz);
  auto ptr_out = key.data_ptr<int64_t>();
  if (key.is_cuda()) {
    xyz2key_gpu((uintk*)ptr_out, (uintk*)ptr_in, num, depth);
  } else {
    xyz2key_cpu((uintk*)ptr_out, (uintk*)ptr_in, num, depth);
  }
  return key;
}

Tensor octree_key2xyz(Tensor key, int depth) {
  key = key.contiguous();  // !!! make sure the Tensor is contiguous
  auto ptr_in = key.data_ptr<int64_t>();
  int num = key.numel();
  CHECK_GE(num, 1) << "The numel of the input tensor must be larger than 1.";

  Tensor xyz = torch::zeros_like(key);
  auto ptr_out = xyz.data_ptr<int64_t>();
  if (key.is_cuda()) {
    key2xyz_gpu((uintk*)ptr_out, (uintk*)ptr_in, num, depth);
  } else {
    key2xyz_cpu((uintk*)ptr_out, (uintk*)ptr_in, num, depth);
  }
  return xyz;
}

Tensor octree_search_key(Tensor key, Tensor octree, int depth, bool key_is_xyz,
                         bool nempty) {
  key = key.contiguous();
  octree = octree.contiguous();  // !!! make sure the Tensor is contiguous
  int64_t* src_key = key.data_ptr<int64_t>();
  int src_h = key.numel();
  CHECK_GE(src_h, 1) << "The numel of the input tensor must be larger than 1.";
  torch::TensorOptions options = key.options();

  Tensor src_key_tmp;
  if (key_is_xyz) {
    src_key_tmp = torch::zeros_like(key);
    int64_t* tmp = src_key_tmp.data_ptr<int64_t>();
    xyz2key_gpu((uintk*)tmp, (uintk*)src_key, src_h, depth);
    src_key = tmp;
  }

  OctreeParser octree_;
  octree_.set_gpu(octree.data_ptr<uint8_t>());
  int des_h = octree_.info().node_num(depth);
  const uintk* des_key = octree_.key_gpu(depth);

  Tensor key_tmp;
  if (nempty) {         // Search the non-empty octree nodes only
    int top_h = des_h;  // cache old des_h
    des_h = octree_.info().node_num_nempty(depth);  // update des_h
    key_tmp = torch::zeros({des_h}, options);
    int64_t* tmp = key_tmp.data_ptr<int64_t>();
    pad_backward_gpu((uintk*)tmp, des_h, 1, des_key, top_h,
                     octree_.children_gpu(depth));
    des_key = (const uintk*)tmp;
  }

  Tensor des_key_tmp;
  if (octree_.info().is_key2xyz()) {
    des_key_tmp = torch::zeros({des_h}, options);
    int64_t* tmp = des_key_tmp.data_ptr<int64_t>();
    xyz2key_gpu((uintk*)tmp, (uintk*)des_key, des_h, depth);
    des_key = (const uintk*)tmp;
  }

  Tensor data_out = torch::zeros({src_h}, options.dtype(torch::kInt32));
  int* ptr_out = data_out.data_ptr<int>();

  // binary search
  search_key_gpu(ptr_out, des_key, des_h, (const uintk*)src_key, src_h);
  return data_out;
}
