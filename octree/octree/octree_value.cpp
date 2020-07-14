#include "octree_value.h"
#include "math_functions.h"

#include <queue>
#include <cmath>

pair<float, float> OctreeValue::fval(const float x, const float y, const float z) const {
  int oct_depth = octree_->info().depth();
  // If it is not an adaptive octree, then set the adp_depth as oct_depth
  int adp_depth = octree_->info().is_adaptive() ?
      octree_->info().adaptive_layer() : oct_depth;

  //pair<node_id, node_depth>
  std::queue<std::pair<int, int> > node_stack;
  node_stack.push(std::make_pair(0, 0));
  float phi = 0.0f, wt = 0.0f;
  while (!node_stack.empty()) {
    // pop
    std::pair<int, int>& node = node_stack.front();
    int id = node.first;
    int depth = node.second;
    int depth_child = depth + 1;
    int id_child = octree_->children_cpu(depth)[id] * 8;
    node_stack.pop();

    // Rescale the input point into the range [1, 2^depth_child]^3
    float scale = 1.0f / float(1 << (oct_depth - depth_child));
    float pos_in[3] = { x * scale, y * scale, z * scale };

    // Deal with the 8 children of the top node
    for (int i = id_child; i < 8 + id_child; ++i) {
      float pos_child[3], nm[3];
      octree_->node_pos(pos_child, i, depth_child);
      float w = weight(pos_in, pos_child);
      if (w != 0) {
        if (octree_->children_cpu(depth_child)[i] < 0 || depth_child == oct_depth) {
          // This node has no children
          if (depth_child >= adp_depth) {
            octree_->node_normal(nm, i, depth_child);
            float len = fabsf(nm[0]) + fabsf(nm[1]) + fabsf(nm[2]);
            if (len != 0) {
              // This node has plane information
              phi += w * basis(pos_in, pos_child, nm);
              wt += w;
            }
          }
        } else {
          // This node has children, so push children
          node_stack.push(std::make_pair(i, depth_child));
        }
      }
    }
  }
  if (wt < 1.0e-4f) wt = 0;
  if (wt != 0) phi /= wt;
  return std::make_pair(phi, wt); //pair<value, weight>
}


float OctreeValue::bspline2(float x) const {
  if (x < -1.5f) {
    return 0;
  } else if (x < -0.5f) {
    return 0.5f * (x + 1.5f) * (x + 1.5f);
  } else if (x < 0.5f) {
    return 0.75f - x * x;
  } else if (x <= 1.5f) {
    return 0.5f * (x - 1.5f) * (x - 1.5f);
  } else {
    return 0;
  }
}

inline float OctreeValue::basis(const float* pos, const float* c, const float* n) const {
  return n[0] * (pos[0] - c[0]) + n[1] * (pos[1] - c[1]) + n[2] * (pos[2] - c[2]);
}

inline float OctreeValue::weight(const float* pos, const float* c) const {
  return bspline2(pos[0] - c[0]) * bspline2(pos[1] - c[1]) * bspline2(pos[2] - c[2]);
}

