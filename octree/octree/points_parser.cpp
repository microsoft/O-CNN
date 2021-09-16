#include "points_parser.h"
#include "math_functions.h"

#include <cstring>
#include <fstream>
#include <chrono>
#include <random>


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

void PointsParser::scale(const float* s) {
  if ((s[0] == 1.0f && s[1] == 1.0f && s[2] == 1.0f) || 
       s[0] == 0.0f || s[1] == 0.0f || s[2] == 0.0f) { return; }

  int npt = info_->pt_num();
  float* pt = this->mutable_points();
  for (int i = 0; i < npt; ++i) {
    int ix3 = i * 3;
    for (int j = 0; j < 3; ++j) {
      pt[ix3 + j] *= s[j];
    }
  }

  if (s[0] == s[1] && s[0] == s[2]) { return; } // uniform scale
  const float t[3] = {1.0f / s[0], 1.0f / s[1], 1.0f / s[2]};

  if (this->info().has_property(PointsInfo::kNormal)) {
    float* nm = this->mutable_normal();
    for (int i = 0; i < npt; ++i) {
      int ix3 = i * 3;
      for (int j = 0; j < 3; ++j) {
        nm[ix3 + j] *= t[j];
      }
    }
    normalize_nx3(nm, npt);
  }
}

void PointsParser::rotate(const float angle, const float* axis) {
  float rot[9];
  rotation_matrix(rot, angle, axis);

  int npt = info_->pt_num();
  vector<float> tmp(3 * npt);
  matrix_prod(tmp.data(), rot, mutable_points(), 3, npt, 3);
  std::copy(tmp.begin(), tmp.end(), mutable_points());

  if (this->info().has_property(PointsInfo::kNormal)) {
    matrix_prod(tmp.data(), rot, this->mutable_normal(), 3, npt, 3);
    std::copy(tmp.begin(), tmp.end(), mutable_normal());
  }
}

void PointsParser::rotate(const float* angles) {
  float rot[9];
  rotation_matrix(rot, angles);

  int npt = info_->pt_num();
  vector<float> tmp(3 * npt);
  matrix_prod(tmp.data(), rot, mutable_points(), 3, npt, 3);
  std::copy(tmp.begin(), tmp.end(), mutable_points());

  if (this->info().has_property(PointsInfo::kNormal)) {
    matrix_prod(tmp.data(), rot, this->mutable_normal(), 3, npt, 3);
    std::copy(tmp.begin(), tmp.end(), mutable_normal());
  }
}

void PointsParser::transform(const float* mat) {
  int npt = info_->pt_num();
  vector<float> tmp(3 * npt);
  matrix_prod(tmp.data(), mat, mutable_points(), 3, npt, 3);
  std::copy(tmp.begin(), tmp.end(), mutable_points());

  if (this->info().has_property(PointsInfo::kNormal)) {
    float mat_it[9];
    inverse_transpose_3x3(mat_it, mat);
    matrix_prod(tmp.data(), mat_it, mutable_normal(), 3, npt, 3);
    bool is_unitary = almost_equal_3x3(mat_it, mat);

    if (!is_unitary) {
      normalize_nx3(tmp.data(), npt);
    }

    std::copy(tmp.begin(), tmp.end(), mutable_normal());
  }
}

vector<int> PointsParser::clip(const float* bbmin, const float* bbmax) {
  int npt = info_->pt_num(), npt_in_bbox = 0;
  float* pts = mutable_points();
  vector<int> in_bbox(npt, 0);
  for (int i = 0; i < npt; ++i) {
    int ix3 = i * 3;
    in_bbox[i] = bbmin[0] < pts[ix3] && pts[ix3] < bbmax[0] &&
        bbmin[1] < pts[ix3 + 1] && pts[ix3 + 1] < bbmax[1] &&
        bbmin[2] < pts[ix3 + 2] && pts[ix3 + 2] < bbmax[2];
    npt_in_bbox += in_bbox[i];
  }

  if (npt_in_bbox == npt) {       // early stop
    return in_bbox;
  }

  if (npt_in_bbox == 0) {         // no points
    // just keep one point to avoid the degenerated case
    npt_in_bbox = 1;
    in_bbox[0] = 1;
    float* p = mutable_points();
    for (int i = 0; i < 3; ++i) { p[i] = bbmin[i]; }
  }

  // discard the points which are out of the bbox
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

  info_->set_pt_num(npt_in_bbox);
  return in_bbox;
}

void PointsParser::add_noise(const float std_pt, const float std_nm) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> dis_pt(0.0f, std_pt), dis_nm(0.0f, std_nm);

  int npt = info_->pt_num();
  if (std_pt > 1.0e-5f) {
    float* pt = mutable_ptr(PointsInfo::kPoint);
    for (int i = 0; i < 3 * npt; ++i) {
      pt[i] += dis_pt(generator);
    }
  }

  if (std_nm > 1.0e-5f && this->info().has_property(PointsInfo::kNormal)) {
    float* nm = mutable_normal();
    for (int i = 0; i < 3 * npt; ++i) {
      nm[i] += dis_nm(generator);
    }
    normalize_nx3(nm, npt);
  }
}

void PointsParser::normalize() {
  float radius, center[3], trans[3];
  bounding_sphere(radius, center, this->points(), this->info().pt_num());
  for (int i = 0; i < 3; ++i) { trans[i] = -center[i]; }

  this->translate(trans);
  this->uniform_scale(1 / (radius + 1.0e-10f));
}

void PointsParser::orient_normal(const string axis) {
  if (!this->info().has_property(PointsInfo::kNormal)) {
    return;  // directly return if there is no normal
  }

  int npt = info_->pt_num();
  float* normal = this->mutable_normal();
  if (axis == "x" || axis == "y" || axis == "z") {
    int j = 0;
    if (axis == "y") j = 1;
    if (axis == "z") j = 2;
    for (int i = 0; i < npt; ++i) {
      int ix3 = i * 3;
      if (normal[ix3 + j] < 0) {
        for (int c = 0; c < 3; ++c) {
          normal[ix3 + c] = -normal[ix3 + c];
        }
      }
    }
  } else if (axis == "xyz") {
    for (int i = 0; i < npt; ++i) {
      int ix3 = i * 3;
      for (int c = 0; c < 3; ++c) {
        if (normal[ix3 + c] < 0) {
          normal[ix3 + c] = -normal[ix3 + c];
        }
      }
    }
  } else {
    // do nothing
  }
}
