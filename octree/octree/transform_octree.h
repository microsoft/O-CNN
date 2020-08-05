#ifndef _OCTREE_TRANSFORM_OCTREES_
#define _OCTREE_TRANSFORM_OCTREES_

#include "octree_parser.h"

class ScanOctree {
 public:
  ScanOctree(float scale = 1.0f) : scale_(scale) {}
  void set_scale(float scale);
  void set_axis(const float* axis, int n = 3);
  void scan(vector<char>& octree_out, const OctreeParser& octree_in,
            const vector<float>& axis);

 protected:
  void bbox_xy(int depth);
  void reset_buffer();
  void z_buffer(vector<int>& drop_flags, const OctreeParser& octree_in);
  void generate_flags(vector<vector<int>>& drop_flags, const OctreeParser& octree_in);
  void trim_octree(vector<char>& octree_out, const OctreeParser& octree_in,
                   vector<vector<int>>& drop_flags);

 protected:
  float scale_;   // scale_ must be large than 0
  int width_;
  vector<int> id_buffer_;
  vector<float> z_buffer_;

  float bbmin_[3], bbmax_[3];
  float x_[3], y_[3], z_[3];
};

void octree_dropout(vector<char>& octree_output, const string& octree_input,
    const int depth_dropout, const float threshold);

void upgrade_key64(vector<char>& octree_out, const vector<char>& octree_in);

#endif // _OCTREE_TRANSFORM_OCTREES_
