#ifndef _OCTREE_CONTOUR_
#define _OCTREE_CONTOUR_

#include <vector>
#include <utility>
#include <unordered_map>

#include "octree_parser.h"
#include "octree_value.h"

using std::pair;
using std::vector;
using std::unordered_map;


class Contour {
 public:
  Contour(const OctreeParser* oct = nullptr): value_(oct) {
    set_octree(oct);
  }
  void set_octree(const OctreeParser* oct) {
    octree_ = oct;
    fval_map_.clear();
  }

  void marching_cube(vector<float>& V, vector<int>& F);

 protected:
  typedef unsigned int uint32;
  pair<float, float> fval(int x, int y, int z);
  bool check_subdividion(const uint32 node_key, const int depth);
  void subdivide(uint32* key_output, const uint32 key_input) const;

 protected:
  OctreeValue value_;
  const OctreeParser* octree_;
  unordered_map<int, pair<float, float> > fval_map_;

};

#endif