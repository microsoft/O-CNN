#ifndef _OCTREE_POINTS_
#define _OCTREE_POINTS_

#include <vector>
#include <string>
#include "points_parser.h"

using std::vector;
using std::string;

class Points: public PointsParser {
 public:
  Points() : buffer_() {}

  // the pts must not be empty, the labels may be empty,
  // the normals & features must not be empty at the same time.
  bool set_points(const vector<float>& pts, const vector<float>& normals,
      const vector<float>& features = vector<float>(),
      const vector<float>& labels = vector<float>());
  void set_points(vector<char>& data); // swap data and buffer_

  const vector<char>& get_buffer() const { return buffer_; }
  
  bool read_points(const string& filename);
  bool write_points(const string& filename) const;
  bool write_ply(const string& filename) const;

 protected:
  vector<char> buffer_;
};


#endif // _OCTREE_POINTS_
