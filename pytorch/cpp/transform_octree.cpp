#include <octree/transform_octree.h>

#include "ocnn.h"

Tensor octree_scan(Tensor octree, vector<float> axis, float scale) {
  // input
  OctreeParser parser;
  parser.set_cpu(octree.data_ptr<uint8_t>());

  // scan
  ScanOctree scan_octree(scale);
  vector<char> octree_out;
  scan_octree.scan(octree_out, parser, axis);

  // output
  torch::TensorOptions options = octree.options();
  Tensor output = torch::zeros(octree_out.size(), options);
  memcpy(output.data_ptr<uint8_t>(), octree_out.data(), octree_out.size());

  return output;
}