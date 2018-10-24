#ifndef CAFFE_UTIL_OCTREE_INFO_
#define CAFFE_UTIL_OCTREE_INFO_

#include <string>

using std::string;

namespace caffe {

class OctreeInfo {
 public:
  enum PropType {
    kKey = 1, kChild = 2, kNeigh = 4, kFeature = 8, kLabel = 16, kSplit = 32
  };
  static const int kPTypeNum = 6;
  static const char kMagicStr[16];

 public:
  OctreeInfo() { reset(); }
  void reset();
  bool check_format(string& msg) const;
  bool is_consistent(const OctreeInfo& info) const;
  bool has_property(PropType ptype) const {
    return (content_flags_ & ptype) != 0;
  }

  int batch_size() const { return batch_size_; }
  int depth() const { return depth_; }
  int full_layer() const { return full_layer_; }
  int adaptive_layer() const { return adp_layer_; }
  float threshold_distance() const { return threshold_dist_; }
  float threshold_normal() const { return threshold_norm_; }
  bool is_adaptive() const { return is_adaptive_; }
  bool has_displace() const { return has_node_dis_; }
  int node_num(int d) const { return nnum_[d]; }
  int node_num_cum(int d) const { return nnum_cum_[d]; }
  int node_num_nempty(int d) const { return nnum_nempty_[d]; }
  int total_nnum() const { return nnum_cum_[depth_ + 1]; }
  int total_nnum_capacity() const { return nnum_cum_[depth_ + 2]; }
  int content_flags() const { return content_flags_; }
  int channel(PropType ptype) const;
  int locations(PropType ptype) const;
  int ptr_dis(PropType ptype, const int depth) const;
  float bbox_max_width() const;
  bool key2xyz() const { return key2xyz_; }
  const float* bbmin() const { return bbmin_; }
  const float* bbmax() const { return bbmax_; }
  int sizeof_octree() const { return ptr_dis_[kPTypeNum]; }

  void set_batch_size(int b);
  void set_depth(int d);
  void set_full_layer(int fd);
  void set_nnum(int d, int num);
  void set_nnum(const int* num);
  void set_nempty(int d, int num);
  void set_nempty(const int* num);
  void set_nnum_cum(int capacity = 0);
  void set_property(PropType ptype, int ch, int lc);
  void set_channel(PropType ptype, int ch);
  void set_location(PropType ptype, int lc);
  void set_ptr_dis();
  void set_bbox(const float* bbmin, const float* bbmax);
  void set_key2xyz(bool b) { key2xyz_ = b; }
  void set_node_dis(bool dis) { has_node_dis_ = dis; }
  void set_adaptive(bool adp) { is_adaptive_ = adp; }
  void set_adaptive_layer(int d) { adp_layer_ = d; }
  void set_threshold_dist(float th) { threshold_dist_ = th; }
  void set_threshold_normal(float th) { threshold_norm_ = th; }

 protected:
  int property_index(PropType ptype) const;

 protected:
  char magic_str_[16];
  int batch_size_;
  int depth_;
  int full_layer_;
  int adp_layer_;
  bool is_adaptive_;
  float threshold_dist_;
  float threshold_norm_;
  bool key2xyz_;
  bool has_node_dis_;   // if true, the last channel of feature is node displacement
  int nnum_[16];        // node number of each depth
  int nnum_cum_[16];    // cumulative node number
  int nnum_nempty_[16]; // non-empty node number of each depth
  int content_flags_;   // indicate the existance of a property
  int channels_[16];    // signal channel
  int locations_[16];   // -1: at all levels; d: at the d^th level
  float bbmin_[3];
  float bbmax_[3];
  char reserved_[256];  // reserved for future usage: 2018/10/31

 private:
  int ptr_dis_[16];
};

} // namespace caffe

#endif // CAFFE_UTIL_OCTREE_INFO_