#ifndef _OCTREE_POINTS_INFO_
#define _OCTREE_POINTS_INFO_

#include <string>

using std::string;

class PointsInfo {
 public:
  enum PropType { kPoint = 1, kNormal = 2, kFeature = 4, kLabel = 8 };
  static const int kPTypeNum = 4;
  static const char kMagicStr[16];

 public:
  PointsInfo() { reset(); }
  void reset();
  bool check_format(string& msg) const;
  bool has_property(PropType ptype) const {
    return (content_flags_ & ptype) != 0;
  }

  int pt_num() const { return pt_num_; }
  int channel(PropType ptype) const;
  int ptr_dis(PropType ptype) const;
  int sizeof_points() const { return ptr_dis_[kPTypeNum]; }

  void set_pt_num(int num) { pt_num_ = num; }
  void set_channel(PropType ptype, const int ch);
  void set_ptr_dis();

 protected:
  int property_index(PropType ptype) const;

 protected:
  char magic_str_[16];
  int pt_num_;
  int content_flags_;
  int channels_[8];
  int ptr_dis_[8];
};

#endif // _OCTREE_POINTS_INFO_
