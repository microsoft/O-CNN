#ifndef _OCTREE_OCTREE_PARSER_
#define _OCTREE_OCTREE_PARSER_

#include <vector>
#include <string>

#include "points.h"
#include "octree_info.h"

using std::vector;
using std::string;

class OctreeParser {
 public:
  typedef unsigned long long uint64;
  typedef unsigned int uint32;
  enum NodeType {kNonEmptyLeaf = -2, kLeaf = -1, kInternelNode = 0 };

 public:
  OctreeParser() : info_(nullptr) {}
  const OctreeInfo& info() const { return *info_; }
  const vector<char>& buffer() const { return buffer_; }
  bool is_empty() const { return info_ == nullptr; }
  NodeType node_type(const int t) const;

  const char* ptr(const OctreeInfo::PropType ptype, const int depth) const;
  const uint32* key(const int depth) const;
  const int* child(const int depth) const;
  const int* neigh(const int depth) const;
  const float* feature(const int depth) const;
  const float* label(const int depth) const;
  const float* split(const int depth) const;

  void set_octree(vector<char>& data); // swap data and buffer_
  void set_octree(const char* data, const int sz);
  void resize_octree(const int sz);
  OctreeInfo& mutable_info() { return *info_; }
  char* mutable_ptr(const OctreeInfo::PropType ptype, const int depth);
  uint32* mutable_key(const int depth);
  int* mutable_child(const int depth);
  int* mutable_neigh(const int depth);
  float* mutable_feature(const int depth);
  float* mutable_label(const int depth);
  float* mutable_split(const int depth);

  bool read_octree(const string& filename);
  bool write_octree(const string& filename) const;

  void octree2pts(Points& point_cloud, int depth_start, int depth_end);
  void octree2mesh(vector<float>& V, vector<int>& F, int depth_start, int depth_end);

 protected:
  // Caveat: for the following to functions, pt and depth
  // must be consistent, i.e pt must be in the range [0, 2^depth]^3
  // compute the key for the sepcified point
  void compute_key(uint32& key, const uint32* pt, const int depth);
  // compute the point coordinate given the key
  void compute_pt(uint32* pt, const uint32& key, const int depth);

  int clamp(int val, int val_min, int val_max);

 protected:
  // the octree is serialized into buffer_
  vector<char> buffer_;
  OctreeInfo* info_;

  // const
  const float ESP = 1.0e-30f;
};

#endif // _OCTREE_OCTREE_