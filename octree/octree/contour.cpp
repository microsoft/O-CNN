#include "contour.h"

#include "marching_cube.h"

void Contour::marching_cube(vector<float>& V, vector<int>& F) {
  // subdivide
  const int depth = octree_->info().depth();
  vector<uintk> nodes_subdivided;
  for (int d = octree_->info().full_layer(); d < depth; ++d) {
    // Check subdividion of octree nodes
    int nnum = octree_->info().node_num(d);
    vector<uintk> nodes_need_subdivide;
    for (int i = 0; i < nnum; ++i) {
      // Only check the leaf nodes
      if (octree_->children_cpu(d)[i] < 0) {
        uintk keyi = octree_->key_cpu(d)[i];
        if (check_subdividion(keyi, d)) {
          nodes_need_subdivide.push_back(keyi);
        }
      }
    }

    // check the subdivided nodes in the last iteration
    for (int i = 0; i < nodes_subdivided.size(); ++i) {
      uintk keyi = nodes_subdivided[i];
      if (check_subdividion(keyi, d)) {
        nodes_need_subdivide.push_back(keyi);
      }
    }

    // subdivide
    size_t sz = nodes_need_subdivide.size();
    nodes_subdivided.resize(8 * sz);
    for (int i = 0; i < sz; ++i) {
      subdivide(nodes_subdivided.data() + 8 * i, nodes_need_subdivide[i]);
    }
  }

  // marching cube
  V.clear();
  F.clear();
  nodes_subdivided.insert(
      nodes_subdivided.end(), octree_->key_cpu(depth),
      octree_->key_cpu(depth) + octree_->info().node_num(depth));
  for (int i = 0; i < nodes_subdivided.size(); ++i) {
    uintk weight_case = 0;
    float corner_val[8], pt[3];
    octree_->key2xyz(pt, nodes_subdivided[i], depth);
    for (int j = 0; j < 8; ++j) {
      int x = pt[0] + MarchingCube::corner_[j][0];
      int y = pt[1] + MarchingCube::corner_[j][1];
      int z = pt[2] + MarchingCube::corner_[j][2];
      auto fvalue = fval(x, y, z);
      corner_val[j] = fvalue.first;
      if (fvalue.second != 0) weight_case |= (1 << j);
    }
    // only consider the voxel that in the support area of the implicit function
    if (weight_case != 255) continue;

    MarchingCube m_cube(corner_val, 0.0f, pt, V.size() / 3);
    m_cube.contouring(V, F);
  }

  // translate and scale points
  if (rescale_) {
    const float* bbmin = octree_->info().bbmin();
    const float scale = octree_->info().bbox_max_width() / float(1 << depth);
    for (int i = 0; i < V.size() / 3; ++i) {
      for (int c = 0; c < 3; ++c) {
        V[i * 3 + c] = V[i * 3 + c] * scale + bbmin[c];
      }
    }
  }
}

pair<float, float> Contour::fval(int x, int y, int z) {
  const int depth = octree_->info().depth();
  int key = (x << 2 * depth) | (y << depth) | z;
  auto it = fval_map_.find(key);
  if (it == fval_map_.end()) {
    auto v = value_.fval(x, y, z);
    fval_map_[key] = v;  // insert new element
    return v;
  } else {
    return it->second;
  }
}

bool Contour::check_subdividion(const uintk node_key, const int depth) {
  // get cooridinates
  int xyz[3] = {0};
  octree_->key2xyz(xyz, node_key, depth);
  int depth_ = octree_->info().depth();
  const int scale = 1 << (depth_ - depth);
  for (int c = 0; c < 3; ++c) {
    xyz[c] *= scale;
  }

  // check 8 cornors
  const uintk mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  uintk cube_case = 0, weight_case = 0;
  for (int i = 0; i < 8; ++i) {
    int x = xyz[0] + MarchingCube::corner_[i][0] * scale;
    int y = xyz[1] + MarchingCube::corner_[i][2] * scale;
    int z = xyz[2] + MarchingCube::corner_[i][2] * scale;

    auto fvalue = fval(x, y, z);  // pair<value, weight>
    if (fvalue.first < 0) cube_case |= mask[i];
    if (fvalue.second != 0) weight_case |= mask[i];
  }
  if (cube_case != 0 && cube_case != 255 && weight_case == 255) return true;

  // check 6 faces
  const int coord[6][3] = {
      {xyz[0], xyz[1], xyz[2]}, {xyz[0] + scale, xyz[1], xyz[2]},
      {xyz[0], xyz[1], xyz[2]}, {xyz[0], xyz[1] + scale, xyz[2]},
      {xyz[0], xyz[1], xyz[2]}, {xyz[0], xyz[1], xyz[2] + scale}};
  const int axis1[6][3] = {{0, 1, 0}, {0, 1, 0}, {1, 0, 0},
                           {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
  const int axis2[6][3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1},
                           {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
  for (int i = 0; i < 6; ++i) {
    for (int m = 0; m < scale; ++m) {
      for (int n = 0; n < scale; ++n) {
        uintk face_case = 0, wt_case = 0;
        for (int k = 0; k < 4; ++k) {
          int m1 = (k & 1) ? m + 1 : m;
          int n1 = (k & 2) ? n + 1 : n;
          int x = coord[i][0] + m1 * axis1[i][0] + n1 * axis2[i][0];
          int y = coord[i][1] + m1 * axis1[i][1] + n1 * axis2[i][1];
          int z = coord[i][2] + m1 * axis1[i][2] + n1 * axis2[i][2];

          auto fvalue = fval(x, y, z);  // pair<value, weight>
          if (fvalue.first < 0) face_case |= mask[k];
          if (fvalue.second != 0) wt_case |= mask[k];
        }
        if (face_case != 0 && face_case != 15 && wt_case == 15) return true;
      }
    }
  }

  return false;
}

void Contour::subdivide(uintk* key_output, const uintk key_input) const {
  typedef typename KeyTrait<uintk>::uints uints;

  if (octree_->info().is_key2xyz()) {
    const uints* pt = reinterpret_cast<const uints*>(&key_input);
    uints x = pt[0] << 1;
    uints y = pt[1] << 1;
    uints z = pt[2] << 1;
    for (int i = 0; i < 8; ++i) {
      uints* xyz = reinterpret_cast<uints*>(key_output + i);
      xyz[0] = (i & 1) ? x + 1 : x;
      xyz[1] = (i & 2) ? y + 1 : y;
      xyz[2] = (i & 4) ? z + 1 : z;
    }
  } else {
    uintk key_in = key_input << 3;
    for (int i = 0; i < 8; ++i) {
      key_output[i] = key_in | i;
    }
  }
}