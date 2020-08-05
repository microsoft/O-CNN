#ifndef _OCTREE_CONTOUR_
#define _OCTREE_CONTOUR_

#include <unordered_map>
#include <utility>
#include <vector>

#include "octree_parser.h"
#include "octree_value.h"
#include "types.h"

using std::pair;
using std::unordered_map;
using std::vector;

class Contour {
 public:
  Contour(const OctreeParser* oct = nullptr, bool rescale = true)
      : value_(oct), rescale_(rescale) {
    set_octree(oct);
  }

  void set_octree(const OctreeParser* oct) {
    octree_ = oct;
    fval_map_.clear();
  }

  void marching_cube(vector<float>& V, vector<int>& F);

 protected:
  pair<float, float> fval(int x, int y, int z);
  bool check_subdividion(const uintk node_key, const int depth);
  void subdivide(uintk* key_output, const uintk key_input) const;

 protected:
  OctreeValue value_;
  const OctreeParser* octree_;
  unordered_map<int, pair<float, float> > fval_map_;
  bool rescale_;
};

#endif