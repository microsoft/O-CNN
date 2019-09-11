#include "points_parser.h"
#include "math_functions.h"

#include <cstring>
#include <fstream>

void PointsParser::set(const void* ptr) {
  const_ptr_ = true;
  metadata_ = reinterpret_cast<char*>(const_cast<void*>(ptr));
  info_ = reinterpret_cast<PointsInfo*>(metadata_);
}

void PointsParser::set(void* ptr, PointsInfo* ptsinfo) {
  const_ptr_ = false;
  metadata_ = reinterpret_cast<char*>(ptr);
  info_ = reinterpret_cast<PointsInfo*>(ptr);
  if (ptsinfo != nullptr) { // update the OctreeInfo with octinfo
    memcpy(info_, ptsinfo, sizeof(PointsInfo));
  }
}

bool PointsParser::is_empty() const {
  return info_ == nullptr || info_->pt_num() == 0;
}

const float* PointsParser::ptr(PointsInfo::PropType ptype) const {
  const float* p = nullptr;
  int dis = info_->ptr_dis(ptype);
  if (-1 != dis) {
    p = reinterpret_cast<const float*>(metadata_ + dis);
  }
  return p;
}

float* PointsParser::mutable_ptr(PointsInfo::PropType ptype) {
  return const_cast<float*>(ptr(ptype));
}

void PointsParser::translate(const float* center) {
  const int dim = 3, npt = info_->pt_num();
  float* pt = mutable_points();
  for (int i = 0; i < npt; ++i) {
    int i3 = i * 3;
    for (int m = 0; m < dim; ++m) {
      pt[i3 + m] += center[m];
    }
  }
}

void PointsParser::displace(const float dis) {
  const int dim = 3, npt = info_->pt_num();
  float* pt = mutable_points();
  float* normal = mutable_normal();
  if (normal == nullptr) return;

  for (int i = 0; i < npt; ++i) {
    int i3 = i * 3;
    for (int m = 0; m < 3; ++m) {
      pt[i3 + m] += normal[i3 + m] * dis;
    }
  }
}

void PointsParser::uniform_scale(const float s) {
  const int npt = info_->pt_num();
  float* pt = mutable_points();
  for (int i = 0; i < 3 * npt; ++i) {
    pt[i] *= s;
  }
}

void PointsParser::rotate(const float angle, const float* axis) {
  float rot[9];
  rotation_matrix(rot, angle, axis);

  int npt = info_->pt_num();
  vector<float> tmp(3 * npt);
  matrix_prod(tmp.data(), rot, mutable_points(), 3, npt, 3);
  copy(tmp.begin(), tmp.end(), mutable_points());

  if (this->info().has_property(PointsInfo::kNormal)) {
    matrix_prod(tmp.data(), rot, this->mutable_normal(), 3, npt, 3);
    copy(tmp.begin(), tmp.end(), mutable_normal());
  }
}

void PointsParser::transform(const float* mat) {
  int npt = info_->pt_num();
  vector<float> tmp(3 * npt);
  matrix_prod(tmp.data(), mat, mutable_points(), 3, npt, 3);
  copy(tmp.begin(), tmp.end(), mutable_points());

  if (this->info().has_property(PointsInfo::kNormal)) {
    float mat_it[9];
    inverse_transpose_3x3(mat_it, mat);
    matrix_prod(tmp.data(), mat_it, mutable_normal(), 3, npt, 3);
    bool is_unitary = almost_equal_3x3(mat_it, mat);

    if (!is_unitary) {
      normalize_nx3(tmp.data(), npt);
    }

    copy(tmp.begin(), tmp.end(), mutable_normal());
  }
}

void PointsParser::clip(const float* bbmin, const float* bbmax) {
  int npt = info_->pt_num(), npt_bbox = 0;
  float* pts = mutable_points();
  vector<int> in_bbox(npt, 0);
  for (int i = 0; i < npt; ++i) {
    int ix3 = i * 3;
    in_bbox[i] = bbmin[0] < pts[ix3] && pts[ix3] < bbmax[0] &&
        bbmin[1] < pts[ix3 + 1] && pts[ix3 + 1] < bbmax[1] &&
        bbmin[2] < pts[ix3 + 2] && pts[ix3 + 2] < bbmax[2];
    npt_bbox += in_bbox[i];
  }

  // Just discard the points which are out of the bbox
  for (int t = 0; t < PointsInfo::kPTypeNum; ++t) {
    auto ptype = static_cast<PointsInfo::PropType>(1 << t);
    int channel = info_->channel(ptype);
    if (channel == 0) { continue; }
    float* p = mutable_ptr(ptype);
    for (int i = 0, j = 0; i < npt; ++i) {
      if (in_bbox[i] == 0) continue;
      int ixC = i * channel, jxC = j * channel;
      for (int c = 0; c < channel; ++c) {
        p[jxC + c] = p[ixC + c];
      }
      j++;
    }
  }

  info_->set_pt_num(npt_bbox);
}