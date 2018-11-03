#include "points.h"

#include <cstring>
#include <fstream>
#include <Miniball.hpp>

#include "util.h"

////////////////////////
const char PtsInfo::kMagicStr[16] = "_POINTS_1.0_";

void PtsInfo::reset() {
  memset(this, 0, sizeof(PtsInfo));
  strcpy(magic_str_, kMagicStr);
}

bool PtsInfo::check_format(string& msg) const {
  msg.clear();
  if (strcmp(kMagicStr, magic_str_) != 0) {
    msg += "The version of points format is not " + string(kMagicStr) + ".\n";
  }
  if (pt_num_ < 0) {
    msg += "The pt_num_ should be larger than 0.\n";
  }
  // todo: add more checks

  // the PtsInfo is valid when no error message is produced
  return msg.empty();
}

int PtsInfo::channel(PropType ptype) const {
  int i = property_index(ptype);
  if (!has_property(ptype)) return 0;
  return channels_[i];
}

void PtsInfo::set_channel(PropType ptype, const int ch) {
  // note: the channel and content_flags_ are consisent.
  // If channels_[i] != 0, then the i^th bit of content_flags_ is 1.
  int i = property_index(ptype);
  channels_[i] = ch;
  content_flags_ |= ptype;
}

int PtsInfo::ptr_dis(PropType ptype) const {
  int i = property_index(ptype);
  if (!has_property(ptype)) return -1;
  return ptr_dis_[i];
}

void PtsInfo::set_ptr_dis() {
  // the accumulated pointer displacement
  ptr_dis_[0] = sizeof(PtsInfo);
  for (int i = 1; i <= kPTypeNum; ++i) { // note the " <= " is used here
    ptr_dis_[i] = ptr_dis_[i - 1] + sizeof(float) * pt_num_ * channels_[i - 1];
  }
}

int PtsInfo::property_index(PropType ptype) const {
  int k = 0, p = ptype;
  for (int i = 0; i < kPTypeNum; ++i) {
    if (0 != (p & (1 << i))) {
      k = i; break;
    }
  }
  return k;
}

////////////////////////
bool Points::read_points(const string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) return false;

  infile.seekg(0, infile.end);
  size_t len = infile.tellg();
  infile.seekg(0, infile.beg);

  buffer_.resize(len);
  infile.read(buffer_.data(), len);
  info_ = reinterpret_cast<PtsInfo*>(buffer_.data());

  infile.close();
  return true;
}

bool Points::write_points(const string& filename) const {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) return false;
  outfile.write(buffer_.data(), buffer_.size());
  outfile.close();
  return true;
}

bool Points::write_ply(const string & filename) const {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) return false;

  // write header
  int n = info_->pt_num();
  outfile << "ply\nformat ascii 1.0\nelement vertex " << n
      << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "property float nx\nproperty float ny\nproperty float nz\n"
      << "element face 0\nproperty list uchar int vertex_indices\n"
      << "end_header" << std::endl;

  // wirte contents
  const int len = 128;
  vector<char> str(n * len, 0);
  char* pstr = str.data();
  const float* pts = ptr(PtsInfo::kPoint);
  const float* normals = ptr(PtsInfo::kNormal);
  for (int i = 0; i < n; ++i) {
    sprintf(pstr + i * len,
        "%.6f %.6f %.6f %.6f %.6f %.6f\n",
        pts[3 * i], pts[3 * i + 1], pts[3 * i + 2],
        normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]);
  }
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = len * i; j < len * (i + 1); ++j) {
      if (str[j] == 0) break;
      str[k++] = str[j];
    }
  }
  outfile.write(str.data(), k);

  outfile.close();
  return false;
}

const float* Points::ptr(PtsInfo::PropType ptype) const {
  const float* p = nullptr;
  int dis = info_->ptr_dis(ptype);
  if (-1 != dis) {
    p = reinterpret_cast<const float*>(buffer_.data() + dis);
  }
  return p;
}

float* Points::mutable_ptr(PtsInfo::PropType ptype) {
  return const_cast<float*>(ptr(ptype));
}

PointsData Points::get_points_data() const
{
    PointsData points_data;

    points_data.npt = info_->pt_num();
    points_data.pts = ptr(PtsInfo::kPoint);
    points_data.normals = ptr(PtsInfo::kNormal);
    points_data.features = ptr(PtsInfo::kFeature);
    points_data.labels = ptr(PtsInfo::kLabel);

    return points_data;
}

PointsBounds Points::get_points_bounds() const
{
    int npt = info_->pt_num();
    const float* pts = ptr(PtsInfo::kPoint);

    PointsBounds bounds;
    bounding_sphere(bounds.radius, bounds.center, pts, npt);

    return bounds;
}

bool Points::set_points(const vector<float>& pts, const vector<float>& normals,
    const vector<float>& features, const vector<float>& labels) {
  /// set info
  int num = pts.size() / 3;
  // !!! Empty input is not allowed
  if (num == 0) return false;
  PtsInfo info;
  info.set_pt_num(num);
  info.set_channel(PtsInfo::kPoint, 3);

  if (!normals.empty()) {
    int c = normals.size() / num;
    int r = normals.size() % num;
    // !!! The channel of normal has to be 3
    if (3 != c || 0 != r) return false;
    info.set_channel(PtsInfo::kNormal, c);
  }

  if (!features.empty()) {
    int c = features.size() / num;
    int r = features.size() % num;
    // !!! The channel of normal has to larger than 0
    if (0 == c || 0 != r) return false;
    info.set_channel(PtsInfo::kFeature, c);
  }

  if (!labels.empty()) {
    int c = labels.size() / num;
    int r = labels.size() % num;
    // !!! The channel of label has to be 1
    if (1 != c || 0 != r) return false;
    info.set_channel(PtsInfo::kLabel, c);
  }

  info.set_ptr_dis();

  /// set buffer
  int sz = info.sizeof_points();
  buffer_.resize(sz);
  memcpy(buffer_.data(), &info, sizeof(PtsInfo));
  info_ = reinterpret_cast<PtsInfo*>(buffer_.data());
  copy(pts.begin(), pts.end(), mutable_ptr(PtsInfo::kPoint));
  if (!normals.empty()) {
    copy(normals.begin(), normals.end(), mutable_ptr(PtsInfo::kNormal));
  }
  if (!features.empty()) {
    copy(features.begin(), features.end(), mutable_ptr(PtsInfo::kFeature));
  }
  if (!labels.empty()) {
    copy(labels.begin(), labels.end(), mutable_ptr(PtsInfo::kLabel));
  }

  return true;
}

void Points::set_points(vector<char>& data) {
  buffer_.swap(data);
  info_ = reinterpret_cast<PtsInfo*>(buffer_.data());
}

//void Points::set_bbox(float* bbmin, float* bbmax) {
//  const int dim = 3;
//  for (int i = 0; i < dim; ++i) {
//    bbmin_[i] = bbmin[i];
//    bbmax_[i] = bbmax[i];
//  }
//}
//
//void Points::set_bbox() {
//  const int dim = 3, npt = info_->pt_num();
//  const float* pt = mutable_ptr(PtsInfo::kPoint);
//  if (npt < 1) {
//    for (int c = 0; c < dim; ++c) {
//      bbmin_[c] = bbmax_[c] = 0.0f;
//    }
//    return;
//  }
//
//  for (int c = 0; c < dim; ++c) {
//    bbmin_[c] = bbmax_[c] = pt[c];
//  }
//  for (int i = 1; i < npt; ++i) {
//    int i3 = i * 3;
//    for (int j = 0; j < dim; ++j) {
//      float tmp = pt[i3 + j];
//      if (tmp < bbmin_[j]) bbmin_[j] = tmp;
//      if (tmp > bbmax_[j]) bbmax_[j] = tmp;
//    }
//  }
//}

void Points::center_about(const float* center) {
  const int dim = 3, npt = info_->pt_num();
  float* pt = mutable_ptr(PtsInfo::kPoint);
  for (int i = 0; i < npt; ++i) {
    int i3 = i * 3;
    for (int m = 0; m < dim; ++m) {
      pt[i3 + m] -= center[m];
    }
  }
}

void Points::displace(const float dis) {
  const int dim = 3, npt = info_->pt_num();
  float* pt = mutable_ptr(PtsInfo::kPoint);
  float* normal = mutable_ptr(PtsInfo::kNormal);
  if (normal == nullptr) return;

  for (int i = 0; i < npt; ++i) {
    int i3 = i * 3;
    for (int m = 0; m < 3; ++m) {
      pt[i3 + m] += normal[i3 + m] * dis;
    }
  }
}

void Points::rotate(const float angle, const float* axis) {
  float rot[9];
  rotation_matrix(rot, angle, axis);

  int npt = info_->pt_num();
  vector<float> tmp(3 * npt);
  matrix_prod(tmp.data(), rot, mutable_ptr(PtsInfo::kPoint), 3, npt, 3);
  copy(tmp.begin(), tmp.end(), mutable_ptr(PtsInfo::kPoint));

  if (this->info().has_property(PtsInfo::kNormal)) {
    matrix_prod(tmp.data(), rot, this->mutable_ptr(PtsInfo::kNormal), 3, npt, 3);
    copy(tmp.begin(), tmp.end(), mutable_ptr(PtsInfo::kNormal));
  }
}

void Points::transform(const float* mat)
{
  int npt = info_->pt_num();
  vector<float> tmp(3 * npt);
  matrix_prod(tmp.data(), mat, mutable_ptr(PtsInfo::kPoint), 3, npt, 3);
  copy(tmp.begin(), tmp.end(), mutable_ptr(PtsInfo::kPoint));

  if (this->info().has_property(PtsInfo::kNormal)) {
    float mat_it[9];
    inverse_transpose_3x3(mat_it, mat);
    matrix_prod(tmp.data(), mat_it, mutable_ptr(PtsInfo::kNormal), 3, npt, 3);
    bool is_unitary = almost_equal_3x3(mat_it, mat);

    if (!is_unitary){
        normalize_nx3(tmp.data(), npt);
    }

    copy(tmp.begin(), tmp.end(), mutable_ptr(PtsInfo::kNormal));
  }
}
