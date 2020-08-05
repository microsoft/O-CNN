#include "merge_octrees.h"

#include "logs.h"
#include "octree_nn.h"
#include "types.h"

void merge_octrees(vector<char>& octree_out, const vector<const char*> octrees_in) {
  MergeOctrees mo;
  mo.init(octrees_in);
  mo.check_input();
  mo.calc_node_num();
  mo.set_batch_info();
  mo.set_batch_parser(octree_out);
  mo.merge_octree();
}

void MergeOctrees::init(const vector<const char*>& octrees) {
  batch_size_ = octrees.size();
  octree_parsers_.resize(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    octree_parsers_[i].set_cpu(octrees[i]);
  }
  depth_ = octree_parsers_[0].info().depth();
  full_layer_ = octree_parsers_[0].info().full_layer();
}

void MergeOctrees::check_input() {
  string err_msg;
  bool valid = octree_parsers_[0].info().check_format(err_msg);
  CHECK(valid) << err_msg;
  for (int i = 1; i < batch_size_; ++i) {
    valid = octree_parsers_[i].info().check_format(err_msg);
    CHECK(valid) << err_msg;
    CHECK(octree_parsers_[0].info().is_consistent(octree_parsers_[i].info()))
        << "The formats of input octrees are not consistent";
  }
}

void MergeOctrees::calc_node_num() {
  // node and non-empty node number in each octree
  int sz = (depth_ + 1) * batch_size_;
  nnum_.resize(sz), nnum_nempty_.resize(sz);
  for (int i = 0; i < batch_size_; ++i) {
    for (int d = 0; d < depth_ + 1; ++d) {
      int p = i * (depth_ + 1) + d;
      nnum_[p] = octree_parsers_[i].info().node_num(d);
      nnum_nempty_[p] = octree_parsers_[i].info().node_num_nempty(d);
    }
  }

  // cumulative node and non-empty node number in each layers
  sz = (depth_ + 1) * (batch_size_ + 1);
  nnum_cum_layer_.resize(sz), nnum_cum_nempty_layer_.resize(sz);
  for (int d = 0; d < depth_ + 1; ++d) {
    nnum_cum_layer_[d] = 0;
    nnum_cum_nempty_layer_[d] = 0;
    for (int i = 0; i < batch_size_; ++i) {
      int p = i * (depth_ + 1) + d;
      int q = p + depth_ + 1;
      nnum_cum_layer_[q] = nnum_[p] + nnum_cum_layer_[p];
      nnum_cum_nempty_layer_[q] = nnum_nempty_[p] + nnum_cum_nempty_layer_[p];
    }
  }

  // cumulative node number for each octree
  sz = (depth_ + 1) * batch_size_;
  nnum_cum_octree_.resize(sz);
  for (int i = 0; i < batch_size_; ++i) {
    nnum_cum_octree_[i * (depth_ + 1)] = 0;
    for (int d = 0; d < depth_; ++d) {
      int p = i * (depth_ + 1) + d;
      nnum_cum_octree_[p + 1] = nnum_cum_octree_[p] + nnum_[p];
    }
  }

  // node and non-empty node number of the batch
  nnum_batch_.resize(depth_ + 1), nnum_nempty_batch_.resize(depth_ + 1);
  for (int d = 0; d < depth_ + 1; ++d) {
    int p = batch_size_ * (depth_ + 1) + d;
    nnum_batch_[d] = nnum_cum_layer_[p];
    nnum_nempty_batch_[d] = nnum_cum_nempty_layer_[p];
  }

  // cumulative node number of the batch
  nnum_cum_batch_.resize(depth_ + 2);
  nnum_cum_batch_[0] = 0;
  for (int d = 0; d < depth_ + 1; ++d) {
    nnum_cum_batch_[d + 1] = nnum_cum_batch_[d] + nnum_batch_[d];
  }
}

void MergeOctrees::set_batch_info() {
  /// set the octinfo
  info_batch_ = octree_parsers_[0].info();
  info_batch_.set_batch_size(batch_size_);
  // add the neighbor property
  const int kNeighChannel = 8;
  info_batch_.set_property(OctreeInfo::kNeigh, kNeighChannel, -1);
  // update nodenumber
  info_batch_.set_nnum(nnum_batch_.data());
  info_batch_.set_nempty(nnum_nempty_batch_.data());
  info_batch_.set_nnum_cum();
  info_batch_.set_ptr_dis();
  // bool valid = info_batch.check_format(err_msg);
  // CHECK(valid) << err_msg;
}

void MergeOctrees::set_batch_parser(vector<char>& octree_out) {
  int sz = info_batch_.sizeof_octree();
  octree_out.resize(sz);
  octbatch_parser_.set_cpu(octree_out.data(), &info_batch_);
}

void MergeOctrees::merge_octree() {
  typedef typename KeyTrait<uintk>::uints uints;

  // omp_set_num_threads(8);
  //#pragma omp parallel for
  for (int i = 0; i < batch_size_; ++i) {
    // copy key
    // the channel and location of key is 1 and -1 
    for (int d = 0; d < depth_ + 1; ++d) {
      if (!info_batch_.has_property(OctreeInfo::kKey)) break;
      int p = i * (depth_ + 1) + d;
      uintk* des = octbatch_parser_.mutable_key_cpu(d) + nnum_cum_layer_[p];
      const uintk* src = octree_parsers_[i].key_cpu(d);
      for (int j = 0; j < nnum_[p]; ++j) {
        des[j] = src[j];
        uints* ptr = reinterpret_cast<uints*>(des + j);
        ptr[3] = i;
      }
    }

    // copy children
    // by default, the channel and location of children is 1 and -1,
    for (int d = 0; d < depth_ + 1; ++d) {
      if (!info_batch_.has_property(OctreeInfo::kChild)) break;
      int p = i * (depth_ + 1) + d;
      int* des = octbatch_parser_.mutable_children_cpu(d) + nnum_cum_layer_[p];
      const int* src = octree_parsers_[i].children_cpu(d);
      for (int j = 0; j < nnum_[p]; ++j) {
        des[j] = -1 == src[j] ? src[j] : src[j] + nnum_cum_nempty_layer_[p];
      }
    }

    // copy data: !NOTE! the type of signal is float!!!
    int feature_channel = info_batch_.channel(OctreeInfo::kFeature);
    int feature_location = info_batch_.locations(OctreeInfo::kFeature);
    int depth_start = feature_location == depth_ ? depth_ : 0;
    for (int d = depth_start; d < depth_ + 1; ++d) {
      if (!info_batch_.has_property(OctreeInfo::kFeature)) break;
      int p = i * (depth_ + 1) + d;
      for (int c = 0; c < feature_channel; c++) {
        float* des = octbatch_parser_.mutable_feature_cpu(d) +
                     c * nnum_batch_[d] + nnum_cum_layer_[p];
        const float* src = octree_parsers_[i].feature_cpu(d) + c * nnum_[p];
        for (int j = 0; j < nnum_[p]; ++j) { des[j] = src[j]; }
      }
    }

    // copy label: !NOTE! the type of label is float!!!
    int label_location = info_batch_.locations(OctreeInfo::kLabel);
    depth_start = label_location == depth_ ? depth_ : 0;
    for (int d = depth_start; d < depth_ + 1; ++d) {
      if (!info_batch_.has_property(OctreeInfo::kLabel)) break;
      int p = i * (depth_ + 1) + d;
      float* des = octbatch_parser_.mutable_label_cpu(d) + nnum_cum_layer_[p];
      const float* src = octree_parsers_[i].label_cpu(d);
      for (int j = 0; j < nnum_[p]; ++j) { des[j] = src[j]; }
    }

    // copy split label: !NOTE! the type of label is float!!!
    int split_location = info_batch_.locations(OctreeInfo::kSplit);
    depth_start = split_location == depth_ ? depth_ : 0;
    for (int d = depth_start; d < depth_ + 1; ++d) {
      if (!info_batch_.has_property(OctreeInfo::kSplit)) break;
      int p = i * (depth_ + 1) + d;
      float* des = octbatch_parser_.mutable_split_cpu(d) + nnum_cum_layer_[p];
      const float* src = octree_parsers_[i].split_cpu(d);
      for (int j = 0; j < nnum_[p]; ++j) des[j] = src[j];
    }
  }

  // calc and set neighbor info
  for (int d = 1; d < depth_ + 1; ++d) {
    if (!info_batch_.has_property(OctreeInfo::kNeigh)) break;
    CHECK(info_batch_.has_property(OctreeInfo::kChild));

    if (d <= full_layer_) {
      calc_neigh_cpu(octbatch_parser_.mutable_neighbor_cpu(d), d, batch_size_);
    } else {
      calc_neigh_cpu(octbatch_parser_.mutable_neighbor_cpu(d),
                     octbatch_parser_.neighbor_cpu(d - 1),
                     octbatch_parser_.children_cpu(d - 1),
                     octbatch_parser_.info().node_num(d - 1));
    }
  }
}
