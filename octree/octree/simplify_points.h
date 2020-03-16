#ifndef _OCTREE_SIMPLIFY_POINTS_
#define _OCTREE_SIMPLIFY_POINTS_

#include <vector>
#include "points.h"

using std::vector;
using std::string;

class SimplifyPoints {
 public:
  SimplifyPoints(int dim, bool avg_points, float offset);
  string set_point_cloud(const string filename);
  bool write_point_cloud(const string filename);
  void simplify();

 protected:
  void transform();
  void inv_transform();

 protected:
  Points point_cloud_;
  float radius_, center_[3];

  int dim_;
  float offset_;
  float offest_obj_;
  bool avg_points_;
  vector<int> spatial_hash_;
};

#endif // _OCTREE_SIMPLIFY_POINTS_
