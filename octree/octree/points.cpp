#include "points.h"
#include "math_functions.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <cassert>

bool Points::read_points(const string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) return false;

  infile.seekg(0, infile.end);
  size_t len = infile.tellg();
  infile.seekg(0, infile.beg);
  if (len < sizeof(PointsInfo)) {
    // the file should at least contain a PtsInfo structure
    infile.close();
    return false;
  }

  buffer_.resize(len);
  char* ptr_ = buffer_.data();
  infile.read(ptr_, len);
  this->set(ptr_, (PointsInfo*)ptr_);

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

  int n = info_->pt_num();
  const float* ptr_pts = points();
  const float* ptr_normal = normal();
  const float* ptr_feature = feature();
  const float* ptr_label = label();
  const int channel_pts = info_->channel(PointsInfo::kPoint);       // 3 channel
  const int channel_normal = info_->channel(PointsInfo::kNormal);   // 3 channel
  const int channel_feature = info_->channel(PointsInfo::kFeature); // x channel
  const int channel_label = info_->channel(PointsInfo::kLabel);     // 1 channel
  // assert(channel_pts == 3 && channel_normal == 3 && channel_label == 1);

  // write header
  std::ostringstream oss;
  oss << "ply\nformat ascii 1.0\nelement vertex " << n
      << "\nproperty float x\nproperty float y\nproperty float z\n";
  if (ptr_normal != nullptr) {
    oss << "property float nx\nproperty float ny\nproperty float nz\n";
  }
  if (ptr_feature != nullptr) {
    for (int i = 0; i < channel_feature; ++i) {
      oss << "property float feature" << i << "\n";
    }
  }
  if (ptr_label != nullptr) {
    oss << "property float label\n";
  }
  oss << "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
  
  // write content
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < channel_pts; ++c) {
      oss << ptr_pts[i*channel_pts + c] << " ";
    }
    for (int c = 0; c < channel_normal; ++c) {
      oss << ptr_normal[i*channel_normal + c] << " ";
    }
    for (int c = 0; c < channel_feature; ++c) {
      oss << ptr_feature[i*channel_feature + c] << " ";
    }
    if (channel_label != 0) {
      oss << ptr_label[i];
    }
    oss << std::endl;
  }

  // write to file
  outfile << oss.str() << std::endl;
  outfile.close();
  return true;
}


bool Points::set_points(const vector<float>& pts, const vector<float>& normals,
    const vector<float>& features, const vector<float>& labels) {
  /// set info
  int num = pts.size() / 3;
  // !!! Empty input is not allowed
  if (num == 0) return false;
  PointsInfo info;
  info.set_pt_num(num);
  info.set_channel(PointsInfo::kPoint, 3);

  if (!normals.empty()) {
    int c = normals.size() / num;
    int r = normals.size() % num;
    // !!! The channel of normal has to be 3
    if (3 != c || 0 != r) return false;
    info.set_channel(PointsInfo::kNormal, c);
  }

  if (!features.empty()) {
    int c = features.size() / num;
    int r = features.size() % num;
    // !!! The channel of feature has to larger than 0
    if (0 == c || 0 != r) return false;
    info.set_channel(PointsInfo::kFeature, c);
  }

  if (!labels.empty()) {
    int c = labels.size() / num;
    int r = labels.size() % num;
    // !!! The channel of label has to be 1
    if (1 != c || 0 != r) return false;
    info.set_channel(PointsInfo::kLabel, c);
  }

  info.set_ptr_dis();

  /// set buffer
  int sz = info.sizeof_points();
  buffer_.resize(sz);
  this->set(buffer_.data(), &info);  // !!! remember to set the point parser !!!
  std::copy(pts.begin(), pts.end(), mutable_ptr(PointsInfo::kPoint));
  if (!normals.empty()) {
    std::copy(normals.begin(), normals.end(), mutable_ptr(PointsInfo::kNormal));
  }
  if (!features.empty()) {
    std::copy(features.begin(), features.end(), mutable_ptr(PointsInfo::kFeature));
  }
  if (!labels.empty()) {
    std::copy(labels.begin(), labels.end(), mutable_ptr(PointsInfo::kLabel));
  }

  return true;
}

void Points::set_points(vector<char>& data) {
  buffer_.swap(data);
  char* ptr_ = buffer_.data();
  this->set(ptr_, (PointsInfo*)ptr_);
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
