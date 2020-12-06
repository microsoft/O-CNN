#ifndef _OCTREE_OCTREE_PARSER_
#define _OCTREE_OCTREE_PARSER_

#include <vector>
#include <string>

#include "types.h"
#include "octree_info.h"

using std::vector;
using std::string;

class OctreeParser {
 public:
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

  const char* ptr_raw_cpu() const { return h_metadata_; }
  const char* ptr_cpu(const PropType ptype, const int depth) const;
  const uintk* key_cpu(const int depth) const;
  const int* children_cpu(const int depth) const;
  const int* neighbor_cpu(const int depth) const;
  const float* feature_cpu(const int depth) const;
  const float* label_cpu(const int depth) const;
  const float* split_cpu(const int depth) const;

  const char* ptr_raw_gpu() const { return d_metadata_; }
  const char* ptr_gpu(const PropType ptype, const int depth) const;
  const uintk* key_gpu(const int depth) const;
  const int* children_gpu(const int depth) const;
  const int* neighbor_gpu(const int depth) const;
  const float* feature_gpu(const int depth) const;
  const float* label_gpu(const int depth) const;
  const float* split_gpu(const int depth) const;

  char* mutable_ptr_cpu(const PropType ptype, const int depth);
  uintk* mutable_key_cpu(const int depth);
  int* mutable_children_cpu(const int depth);
  int* mutable_neighbor_cpu(const int depth);
  float* mutable_feature_cpu(const int depth);
  float* mutable_label_cpu(const int depth);
  float* mutable_split_cpu(const int depth);

  char* mutable_ptr_gpu(const PropType ptype, const int depth);
  uintk* mutable_key_gpu(const int depth);
  int* mutable_children_gpu(const int depth);
  int* mutable_neighbor_gpu(const int depth);
  float* mutable_feature_gpu(const int depth);
  float* mutable_label_gpu(const int depth);
  float* mutable_split_gpu(const int depth);

  //////////////////////////////////////
  void node_pos(float* xyz, int id, int depth, float* xyz_base = nullptr,
                bool clp = false) const;
  void node_normal(float* n, int id, int depth) const;
  float node_dis(int id, int depth) const;
  template<typename Dtype>
  void key2xyz(Dtype* xyz, const uintk& key, const int depth) const;

 protected:
  // original pointer
  char* h_metadata_;
  char* d_metadata_;
  OctreeInfo* info_;
  bool const_ptr_;

 private:
  OctreeInfo info_buffer_;
};

#endif // _OCTREE_OCTREE_PARSER_
