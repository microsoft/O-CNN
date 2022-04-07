#ifndef KEY64
#define KEY64
#endif
#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

namespace {

Tensor octree_property_gpu(Tensor octree_in, string property, int depth) {
  OctreeParser octree_;
  octree_.set_gpu(octree_in.data_ptr<uint8_t>());

  int octree_depth = octree_.info().depth();
  int node_num = octree_.info().node_num(depth);
  int total_node_num = octree_.info().total_nnum();
  int nnum = depth > 0 ? node_num : total_node_num;

  torch::TensorOptions options = octree_in.options();
  Tensor data_out = torch::zeros({1}, options);

  if (property == "key") {
    const uintk* ptr = octree_.key_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    uintk* des_ptr = (uintk*)data_out.data_ptr<int64_t>();
    if (octree_.info().is_key2xyz()) {
      if (depth > 0) {
        xyz2key_gpu(des_ptr, ptr, total_num, depth);
      } else {
        for (int d = 1; d < octree_depth + 1; d++) {
          int nnum_d = octree_.info().node_num(d);
          int ncum_d = octree_.info().node_num_cum(d);
          xyz2key_gpu(des_ptr + ncum_d, ptr + ncum_d, nnum_d, d);
        }
      }
    } else {
      memcpy_gpu(total_num, ptr, des_ptr);
    }
  }

  else if (property == "xyz") {
    const uintk* ptr = octree_.key_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    uintk* des_ptr = (uintk*)data_out.data_ptr<int64_t>();
    if (!octree_.info().is_key2xyz()) {
      if (depth > 0) {
        key2xyz_gpu(des_ptr, ptr, total_num, depth);
      } else {
        for (int d = 1; d < octree_depth + 1; d++) {
          int nnum_d = octree_.info().node_num(d);
          int ncum_d = octree_.info().node_num_cum(d);
          key2xyz_gpu(des_ptr + ncum_d, ptr + ncum_d, nnum_d, d);
        }
      }
    } else {
      memcpy_gpu(total_num, ptr, des_ptr);
    }
  }

  else if (property == "index") {
    const uintk* key_ptr = octree_.key_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    key2idx_gpu(data_out.data_ptr<int>(), key_ptr, total_num);
  }

  else if (property == "child") {
    const int* child_ptr = octree_.children_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kChild);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    memcpy_gpu(total_num, child_ptr, data_out.data_ptr<int>());
  }

  else if (property == "neigh") {
    const int* neigh_ptr = octree_.neighbor_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kNeigh);
    int total_num = channel * nnum;
    data_out = torch::zeros({nnum, channel}, options.dtype(torch::kInt32));
    memcpy_gpu(total_num, neigh_ptr, data_out.data_ptr<int>());
  }

  else if (property == "feature") {
    const float* feature_ptr = octree_.feature_gpu(depth);
    CHECK(feature_ptr != nullptr) << "The features do not exist: d = " << depth;
    int channel = octree_.info().channel(OctreeInfo::kFeature);
    int total_num = channel * nnum;
    data_out =
        torch::zeros({1, channel, nnum, 1}, options.dtype(torch::kFloat32));
    memcpy_gpu(total_num, feature_ptr, data_out.data_ptr<float>());
  }

  else if (property == "label") {
    const float* label_ptr = octree_.label_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kLabel);
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_gpu(total_num, label_ptr, data_out.data_ptr<float>());
  }

  else if (property == "split") {
    const float* split_ptr = octree_.split_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kSplit);
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_gpu(total_num, split_ptr, data_out.data_ptr<float>());
  }

  else if (property == "node_num") {
    int num = depth > 0 ? 1 : octree_depth + 1;
    data_out = torch::zeros({num}, options.dtype(torch::kInt32));
    const int* ptr = octree_.info().node_num_ptr();
    memcpy_gpu(num, ptr + depth, data_out.data_ptr<int>());
  }

  else if (property == "node_num_ne" || property == "node_num_nempty") {
    int num = depth > 0 ? 1 : octree_depth + 1;
    data_out = torch::zeros({num}, options.dtype(torch::kInt32));
    const int* ptr = octree_.info().node_nempty_ptr();
    memcpy_gpu(num, ptr + depth, data_out.data_ptr<int>());
  }

  else if (property == "node_num_cum") {
    int num = depth > 0 ? 1 : octree_depth + 2;
    const int* ptr = octree_.info().node_num_cum_ptr();
    data_out = torch::zeros({num}, options.dtype(torch::kInt32));
    memcpy_gpu(num, ptr + depth, data_out.data_ptr<int>());
  }

  else if (property == "batch_size") {
    int batch_size = octree_.info().batch_size();
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_gpu(1, &batch_size, data_out.data_ptr<int>());
  }

  else if (property == "depth") {
    int depth = octree_.info().depth();
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_gpu(1, &depth, data_out.data_ptr<int>());
  }

  else if (property == "full_depth") {
    int full_depth = octree_.info().full_layer();
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_gpu(1, &full_depth, data_out.data_ptr<int>());
  }

  else {
    LOG(FATAL) << "Unsupport octree property: " << property;
  }

  return data_out;
}

Tensor octree_property_cpu(Tensor octree_in, string property, int depth) {
  OctreeParser octree_;
  octree_.set_cpu(octree_in.data_ptr<uint8_t>());

  int octree_depth = octree_.info().depth();
  int node_num = octree_.info().node_num(depth);
  int total_node_num = octree_.info().total_nnum();
  int nnum = depth > 0 ? node_num : total_node_num;

  torch::TensorOptions options = octree_in.options();
  Tensor data_out = torch::zeros({1}, options);

  if (property == "key") {
    const uintk* ptr = octree_.key_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    uintk* des_ptr = (uintk*)data_out.data_ptr<int64_t>();
    if (octree_.info().is_key2xyz()) {
      if (depth > 0) {
        xyz2key_cpu(des_ptr, ptr, total_num, depth);
      } else {
        for (int d = 1; d < octree_depth + 1; d++) {
          int nnum_d = octree_.info().node_num(d);
          int ncum_d = octree_.info().node_num_cum(d);
          xyz2key_cpu(des_ptr + ncum_d, ptr + ncum_d, nnum_d, d);
        }
      }
    } else {
      memcpy_cpu(total_num, ptr, des_ptr);
    }

  }

  else if (property == "xyz") {
    const uintk* ptr = octree_.key_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    uintk* des_ptr = (uintk*)data_out.data_ptr<int64_t>();
    if (!octree_.info().is_key2xyz()) {
      if (depth > 0) {
        key2xyz_cpu(des_ptr, ptr, total_num, depth);
      } else {
        for (int d = 1; d < octree_depth + 1; d++) {
          int nnum_d = octree_.info().node_num(d);
          int ncum_d = octree_.info().node_num_cum(d);
          key2xyz_cpu(des_ptr + ncum_d, ptr + ncum_d, nnum_d, d);
        }
      }
    } else {
      memcpy_cpu(total_num, ptr, des_ptr);
    }
  }

  else if (property == "index") {
    const uintk* key_ptr = octree_.key_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    key2idx_cpu(data_out.data_ptr<int>(), key_ptr, total_num);
  }

  else if (property == "child") {
    const int* child_ptr = octree_.children_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kChild);  // = 1
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    memcpy_cpu(total_num, child_ptr, data_out.data_ptr<int>());
  }

  else if (property == "neigh") {
    const int* neigh_ptr = octree_.neighbor_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kNeigh);
    int total_num = channel * nnum;
    data_out = torch::zeros({nnum, channel}, options.dtype(torch::kInt32));
    memcpy_cpu(total_num, neigh_ptr, data_out.data_ptr<int>());
  }

  else if (property == "feature") {
    const float* feature_ptr = octree_.feature_cpu(depth);
    CHECK(feature_ptr != nullptr) << "The features do not exist: d = " << depth;
    int channel = octree_.info().channel(OctreeInfo::kFeature);
    int total_num = channel * nnum;
    data_out =
        torch::zeros({1, channel, nnum, 1}, options.dtype(torch::kFloat32));
    memcpy_cpu(total_num, feature_ptr, data_out.data_ptr<float>());
  }

  else if (property == "label") {
    const float* label_ptr = octree_.label_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kLabel);
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_cpu(total_num, label_ptr, data_out.data_ptr<float>());
  }

  else if (property == "split") {
    const float* split_ptr = octree_.split_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kSplit);
    int total_num = channel * nnum;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_cpu(total_num, split_ptr, data_out.data_ptr<float>());
  }

  else if (property == "node_num") {
    int num = depth > 0 ? 1 : octree_depth + 1;
    data_out = torch::zeros({num}, options.dtype(torch::kInt32));
    const int* ptr = octree_.info().node_num_ptr();
    memcpy_cpu(num, ptr + depth, data_out.data_ptr<int>());
  }

  else if (property == "node_num_ne" || property == "node_num_nempty") {
    int num = depth > 0 ? 1 : octree_depth + 1;
    data_out = torch::zeros({num}, options.dtype(torch::kInt32));
    const int* ptr = octree_.info().node_nempty_ptr();
    memcpy_cpu(num, ptr + depth, data_out.data_ptr<int>());
  }

  else if (property == "node_num_cum") {
    int num = depth > 0 ? 1 : octree_depth + 2;
    const int* ptr = octree_.info().node_num_cum_ptr();
    data_out = torch::zeros({num}, options.dtype(torch::kInt32));
    memcpy_cpu(num, ptr + depth, data_out.data_ptr<int>());
  }

  else if (property == "batch_size") {
    int batch_size = octree_.info().batch_size();
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_cpu(1, &batch_size, data_out.data_ptr<int>());
  }

  else if (property == "depth") {
    int depth = octree_.info().depth();
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_cpu(1, &depth, data_out.data_ptr<int>());
  }

  else if (property == "full_depth") {
    int full_depth = octree_.info().full_layer();
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_cpu(1, &full_depth, data_out.data_ptr<int>());
  }

  else {
    LOG(FATAL) << "Unsupport octree property: " << property;
  }

  return data_out;
}

Tensor octree_set_property_gpu(Tensor octree_in, Tensor data_in, int depth) {
  Tensor octree_out = octree_in.clone();

  OctreeParser octree_;
  octree_.set_gpu(octree_out.data_ptr<uint8_t>());
  float* property_ptr = octree_.mutable_feature_gpu(depth);

  int length = octree_.info().node_num(depth);
  int channel = octree_.info().channel(OctreeInfo::kFeature);
  int count = length * channel;
  data_in = data_in.contiguous();
  CHECK_EQ(count, data_in.numel()) << "Wrong Property Size";
  memcpy_gpu(count, data_in.data_ptr<float>(), property_ptr);

  return octree_out;
}

}  // anonymous namespace

// API implementation
Tensor octree_property(Tensor octree_in, string property, int depth) {
  if (octree_in.is_cuda()) {
    return octree_property_gpu(octree_in, property, depth);
  } else {
    return octree_property_cpu(octree_in, property, depth);
  }
}

Tensor octree_set_property(Tensor octree_in, Tensor data_in, int depth) {
  CHECK(octree_in.is_cuda());
  CHECK(data_in.is_cuda());
  return octree_set_property_gpu(octree_in, data_in, depth);
}