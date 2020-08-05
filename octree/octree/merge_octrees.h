#ifndef _OCTREE_MERGE_OCTREES_
#define _OCTREE_MERGE_OCTREES_

#include <vector>

#include "octree_parser.h"

using std::vector;

void merge_octrees(vector<char>& octree_out, const vector<const char*> octrees_in);

// A simple implementation of index matrix
class Index {
 public:
  Index(int col, int row) { reset(col, row); }

  void reset(int col, int row) {
    row_ = row;
    col_ = col;
    data_.assign(row_ * col_, 0);
  }

  int operator()(int c, int r) const { return data_[row_ * c + r]; }

  int& operator()(int c, int r) { return data_[row_ * c + r]; }

 protected:
  int row_, col_;
  vector<int> data_;
};

class MergeOctrees {
 public:
  void init(const vector<const char*>& octrees);
  void check_input();
  void calc_node_num();
  void set_batch_info();
  void set_batch_parser(vector<char>& octree_out);
  void merge_octree();

 private:
  int depth_;
  int full_layer_;
  int batch_size_;
  vector<OctreeParser> octree_parsers_;

  vector<int> nnum_;
  vector<int> nnum_nempty_;
  vector<int> nnum_cum_layer_;
  vector<int> nnum_cum_nempty_layer_;
  vector<int> nnum_cum_octree_;
  vector<int> nnum_batch_;
  vector<int> nnum_nempty_batch_;
  vector<int> nnum_cum_batch_;

  OctreeInfo info_batch_;
  OctreeParser octbatch_parser_;
};

#endif  // _OCTREE_MERGE_OCTREES_
