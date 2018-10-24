#ifndef CAFFE_UTIL_OCTREE_PARSER_
#define CAFFE_UTIL_OCTREE_PARSER_

#include <vector>
#include <string>

#include "caffe/util/octree_info.hpp"

using std::vector;
using std::string;

namespace caffe {

class OctreeParser {
 public:
  typedef unsigned int uint32;
  typedef unsigned long long uint64;
  typedef OctreeInfo::PropType PropType;
  enum NodeType {kNonEmptyLeaf = -2, kLeaf = -1, kInternelNode = 0 };

 public:
  OctreeParser() : h_metadata_(nullptr), d_metadata_(nullptr),
    info_(nullptr), const_ptr_(true), info_buffer_() {}
  void set_cpu(const void* ptr);
  void set_gpu(const void* ptr, const void* oct_info = nullptr);
  void set_cpu(void* ptr, OctreeInfo* octinfo = nullptr);
  void set_gpu(void* ptr, OctreeInfo* octinfo = nullptr);

  const OctreeInfo& info() const { return *info_; }
  OctreeInfo& mutable_info() { return *info_; }

  NodeType node_type(const int t) const;
  bool is_empty() const { return info_ == nullptr; }

  const char* ptr_cpu(const PropType ptype, const int depth) const;
  const unsigned int* key_cpu(const int depth) const;
  const int* children_cpu(const int depth) const;
  const int* neighbor_cpu(const int depth) const;
  const float* feature_cpu(const int depth) const;
  const float* label_cpu(const int depth) const;
  const float* split_cpu(const int depth) const;

  const char* ptr_gpu(const PropType ptype, const int depth) const;
  const unsigned int* key_gpu(const int depth) const;
  const int* children_gpu(const int depth) const;
  const int* neighbor_gpu(const int depth) const;
  const float* feature_gpu(const int depth) const;
  const float* label_gpu(const int depth) const;
  const float* split_gpu(const int depth) const;

  char* mutable_ptr_cpu(const PropType ptype, const int depth);
  unsigned int* mutable_key_cpu(const int depth);
  int* mutable_children_cpu(const int depth);
  int* mutable_neighbor_cpu(const int depth);
  float* mutable_feature_cpu(const int depth);
  float* mutable_label_cpu(const int depth);
  float* mutable_split_cpu(const int depth);

  char* mutable_ptr_gpu(const PropType ptype, const int depth);
  unsigned int* mutable_key_gpu(const int depth);
  int* mutable_children_gpu(const int depth);
  int* mutable_neighbor_gpu(const int depth);
  float* mutable_feature_gpu(const int depth);
  float* mutable_label_gpu(const int depth);
  float* mutable_split_gpu(const int depth);

 protected:
  // Caveat: for the following to functions, pt and depth
  // must be consistent, i.e pt must be in the range [0, 2^depth]^3
  // compute the key for the sepcified point
  void compute_key(uint32& key, const uint32* pt, const int depth);
  // compute the point coordinate given the key
  void compute_pt(uint32* pt, const uint32& key, const int depth);

 protected:
  // original pointer
  char* h_metadata_;
  char* d_metadata_;
  OctreeInfo* info_;
  bool const_ptr_;

 private:
  OctreeInfo info_buffer_;
};

} // namespace caffe

#endif // CAFFE_UTIL_OCTREE_PARSER_