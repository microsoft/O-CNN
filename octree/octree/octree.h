#ifndef _OCTREE_OCTREE_
#define _OCTREE_OCTREE_

#include <vector>
#include <string>

#include "points.h"
#include "octree_info.h"
#include "octree_parser.h"

using std::vector;
using std::string;

class Octree : public OctreeParser {
 public:
  Octree() {}
  void set_octree(const Octree& octree_in); // copy from octree_in
  void set_octree(vector<char>& data);      // swap data and buffer_
  void set_octree(const char* data, const int sz);
  void resize_octree(const int sz);
  const vector<char>& buffer() const { return buffer_; }

  bool read_octree(const string& filename);
  bool write_octree(const string& filename) const;
  string get_binary_string() const;

  void build(const OctreeInfo& octree_info, const Points& point_cloud);
  void trim_octree();

  // serialize the results of the function build() into the buffer_
  void serialize();
  bool save_legacy(string& filename);

  void octree2pts(Points& point_cloud, int depth_start, int depth_end, bool rescale = true) const;
  void octree2mesh(vector<float>& V, vector<int>& F, int depth_start, int depth_end) const;

 protected:
  void clear(int depth = 0);
  void normalize_pts(vector<float>& pts_scaled, const Points& pts);
  void sort_keys(vector<uint32>& sorted_keys, vector<uint32>& sorted_idx,
      const vector<float>& pts_scaled);
  void unique_key(vector<uint32>& node_key, vector<uint32>& pidx);

  void build_structure(vector<uint32>& node_keys);
  void calc_node_num();  // called after the function build_structure()

  void calc_signal(const Points& point_cloud, const vector<float>& pts_scaled,
      const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx);
  void calc_signal(const bool calc_normal_err, const bool calc_dist_err);
  void extrapolate_signal();

  void key_to_xyz(vector<vector<uint32> >& xyz);
  void calc_split_label();

  template<typename Dtype>
  void serialize(Dtype* des, const vector<vector<Dtype> >& src, const int location);

  void covered_depth_nodes();
  void valid_depth_range(int& depth_start, int& depth_end) const;

 protected:
  // the octree is serialized into buffer_
  vector<char> buffer_;

  // Note: structure of arrays(SoA), instead of array of structures(AoS), is
  // adopted. The reason is that the SoA is more friendly to GPU-based implementation.
  // In the future, probably I will try to implement this class with CUDA accroding
  // to this CPU-based code.
  vector<vector<uint32> > keys_;
  vector<vector<int> > children_;

  vector<vector<float> > displacement_;
  // split_label: 0 - empty; 1 - non-empty, split; 2 - surface-well-approximated
  vector<vector<float> > split_labels_;

  vector<vector<float> > avg_normals_;   // 3 x N matrix
  vector<vector<float> > avg_features_;  // 3 x N matrix
  vector<vector<float> > avg_pts_;       // 3 x N matrix
  vector<vector<float> > avg_labels_;
  int max_label_;

  OctreeInfo oct_info_;

  // the node number and starting index of depth layer node covered
  vector<vector<int> > dnum_;
  vector<vector<int> > didx_;
  vector<vector<float> > normal_err_;
  vector<vector<float> > distance_err_;
};

#endif // _OCTREE_OCTREE_