#ifndef _OCTREE_OCTREE_VALUE_
#define _OCTREE_OCTREE_VALUE_

#include <utility>
#include "octree_parser.h"

using std::pair;

class OctreeValue {
 public:
  OctreeValue(const OctreeParser* oct = nullptr) : octree_(oct) {}
  pair<float, float> fval(const float x, const float y, const float z) const;

 protected:
  float bspline2(float x) const;
  inline float weight(const float* pos, const float* c) const;
  inline float basis(const float* pos, const float* c, const float* n) const;

 protected:
  const OctreeParser* octree_;
};

#endif // _OCTREE_OCTREE_VALUE_
