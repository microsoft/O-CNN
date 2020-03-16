#ifndef _OCTREE_TRANSFORM_POINTS_
#define _OCTREE_TRANSFORM_POINTS_

#include "points.h"

class DropPoints {
 public:
  DropPoints(int dim, float ratio, const float* bbmin, const float* bbmax);
  void dropout(Points& points);

 protected:
  int hash(const float* pt);

 protected:
  int dim_;
  float ratio_;
  float bbmin_[3], bbmax_[3], iwidth_[3];
  vector<int> spatial_hash_;
};


#endif // _OCTREE_TRANSFORM_POINTS_