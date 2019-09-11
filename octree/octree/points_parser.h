#ifndef _OCTREE_POINTS_PARSER_
#define _OCTREE_POINTS_PARSER_

#include <vector>
#include <string>
#include "points_info.h"

using std::vector;
using std::string;

class PointsParser {
 public:
  PointsParser() : metadata_(nullptr), info_(nullptr), const_ptr_(true)  {}
  void set(const void* ptr);
  void set(void* ptr, PointsInfo* ptsinfo = nullptr);
  bool is_empty() const;

  const PointsInfo& info() const { return *info_; }
  PointsInfo& mutable_info() { return *info_; }

  const float* ptr(PointsInfo::PropType ptype) const;
  float* mutable_ptr(PointsInfo::PropType ptype);

  const float* points() const { return ptr(PointsInfo::kPoint); }
  const float* normal() const { return ptr(PointsInfo::kNormal); }
  const float* feature() const { return ptr(PointsInfo::kFeature); }
  const float* label() const { return ptr(PointsInfo::kLabel); }
  float* mutable_points() { return mutable_ptr(PointsInfo::kPoint); }
  float* mutable_normal() { return mutable_ptr(PointsInfo::kNormal); }
  float* mutable_feature() { return mutable_ptr(PointsInfo::kFeature); }
  float* mutable_label() { return mutable_ptr(PointsInfo::kLabel); }

  // todo: move the following functions out of this class
  void translate(const float* center);
  void displace(const float dis);
  void uniform_scale(const float s);
  void rotate(const float angle, const float* axis); // angle in radian
  void transform(const float* trans_matrix);
  void clip(const float* bbmin, const float* bbmax);


 protected:
  char* metadata_;
  PointsInfo* info_;
  bool const_ptr_;

 private:
  PointsInfo info_buffer_;
};


#endif // _OCTREE_POINTS_PARSER_
