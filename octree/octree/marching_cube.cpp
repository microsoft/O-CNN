#include "marching_cube.h"


inline int MarchingCube::btwhere(int x) const {
  float f = (unsigned int)x;
  return ((*(unsigned int*)(&f)) >> 23) - 127;
}

inline void MarchingCube::interpolation(float* pt, const float* pt1,
    const float* pt2, const float f1, const float f2) const {
  float df = f2 - f1;
  if (df == 0) df += 1.0e-10f;
  float t = -f1 / df;
  for (int i = 0; i < 3; ++i) {
    pt[i] = pt1[i] + t * (pt2[i] - pt1[i]);
  }
}

MarchingCube::MarchingCube(const float* fval, float iso_val, const float* left_btm,
    int vid) {
  set(fval, iso_val, left_btm, vid);
}

void MarchingCube::set(const float* fval, float iso_val, const float* left_btm,
    int vid) {
  fval_ = fval;
  iso_value_ = iso_val;
  left_btm_ = left_btm;
  vtx_id_ = vid;
}

unsigned int MarchingCube::compute_cube_case() const {
  const unsigned int mask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
  unsigned int cube_case = 0;
  for (int i = 0; i < 8; ++i) {
    if (fval_[i] < iso_value_) cube_case |= mask[i];
  }
  return cube_case;
}

void MarchingCube::contouring(vector<float>& vtx, vector<int>& face) {
  // compute cube cases
  unsigned int cube_case = compute_cube_case();

  // generate vtx
  int vid[12], id = vtx_id_;
  int edge_mask = edge_table_[cube_case];
  while (edge_mask != 0) {
    int pos = btwhere(edge_mask & (-edge_mask));

    // set vertex id
    vid[pos] = id++;

    // calc points
    float pti[3];
    int v1 = edge_vert_[pos][0];
    int v2 = edge_vert_[pos][1];
    interpolation(pti, corner_[v1], corner_[v2], fval_[v1], fval_[v2]);
    for (int j = 0; j < 3; ++j) {
      float p = pti[j] + left_btm_[j];
      vtx.push_back(p);
    }

    edge_mask &= edge_mask - 1;
  }

  // generate triangle
  const int* tri = tri_table_[cube_case];
  for (int i = 0; i < 16; ++i) {
    if (tri[i] == -1) break;
    face.push_back(vid[tri[i]]);
  }
}

void MarchingCube::compute(const vector<float>& fval, float iso_val,
                           const vector<float>& left_btm, int vid) {
  vtx_.clear();
  face_.clear();
  set(fval.data(), iso_val, left_btm.data(), vid);
  contouring(vtx_, face_);
}

void intersect_cube(vector<float>& V, const float* pt, const float* pt_base,
    const float* normal) {
  // compute f_val
  float fval[8] = { 0 };
  for (int k = 0; k < 8; ++k) {
    for (int j = 0; j < 3; ++j) {
      fval[k] += (MarchingCube::corner_[k][j] + pt_base[j] - pt[j]) * normal[j];
    }
  }

  // marching cube
  V.clear();
  vector<int> F;
  MarchingCube mcube(fval, 0, pt_base, 0);
  mcube.contouring(V, F);
}


void marching_cube_octree(vector<float>& V, vector<int>& F, const vector<float>& pts,
    const vector<float>& pts_ref, const vector<float>& normals) {
  int num = pts.size() / 3;
  V.clear(); F.clear();
  for (int i = 0; i < num; ++i) {
    // get point and normal
    int ix3 = i * 3;
    float pt[3], pt_ref[3], normal[3];
    for (int j = 0; j < 3; ++j) {
      pt_ref[j] = pts_ref[ix3 + j];      // the reference point
      pt[j] = pts[ix3 + j] - pt_ref[j];  // the local displacement
      normal[j] = normals[ix3 + j];
    }

    // compute f_val
    float fval[8] = {0};
    for (int k = 0; k < 8; ++k) {
      for (int j = 0; j < 3; ++j) {
        fval[k] += (MarchingCube::corner_[k][j] - pt[j]) * normal[j];
      }
    }

    // marching cube
    int vid = V.size() / 3;
    MarchingCube mcube(fval, 0, pt_ref, vid);
    mcube.contouring(V, F);
  }
}

