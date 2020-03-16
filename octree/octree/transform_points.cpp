#include "transform_points.h"
#include <random>
#include <ctime>

DropPoints::DropPoints(int dim, float ratio, const float* bbmin, const float* bbmax) {
  dim_ = dim;
  ratio_ = ratio;
  for (int i = 0; i < 3; ++i) {
    bbmin_[i] = bbmin[i];
    bbmax_[i] = bbmax[i];
    iwidth_[i] = (float) dim / (bbmax[i] - bbmin[i] + 1.0e-10f);
  }
}

int DropPoints::hash(const float* pt) {
  float xyz[3] = { 0 };
  for (int i = 0; i < 3; ++i) {
    xyz[i] = static_cast<int>((pt[i] - bbmin_[i]) * iwidth_[i]);
  }
  int h = (xyz[0] * dim_ + xyz[1]) * dim_ + xyz[2];
  return h;
}

void DropPoints::dropout(Points& points) {
  if (ratio_ < 1.0e-5f) return; // trival case
  std::default_random_engine generator(static_cast<unsigned>(time(nullptr)));
  std::bernoulli_distribution distribution(ratio_);

  const int hash_num = dim_ * dim_ * dim_;
  spatial_hash_.assign(hash_num, -1);
  int pt_num = points.info().pt_num(), pt_num_remain = 0;
  vector<int> pt_flags(pt_num, -1); // -1 - uninitialized, 0 - keep, 1 - drop
  const float* pts = points.points();

  for (int i = 0, id = 0; i < pt_num; ++i) {
    int h = hash(pts + 3 * i);
    if (h >= hash_num) { h = h % hash_num; }
    int hash_val = spatial_hash_[h];
    if (hash_val == -1) {
      hash_val = distribution(generator);
      spatial_hash_[h] = hash_val;
    }
    pt_flags[i] = hash_val;
    if (hash_val == 0) { pt_num_remain++; }
  }

  // keep at least one point
  if (pt_num_remain == 0) {
    pt_num_remain = 1;
    pt_flags[0] = 0;
  }

  // lazy delete: just move the content, do not re-allocate memory
  for (int p = 0; p < PointsInfo::kPTypeNum; ++p) {
    auto ptype = static_cast<PointsInfo::PropType>(1 << p);
    float* ptr = points.mutable_ptr(ptype);
    if (ptr == nullptr) continue;
    int ch = points.info().channel(ptype);
    for (int i = 0, j = 0; i < pt_num; ++i) {
      if (pt_flags[i] == 1) continue;
      for (int c = 0; c < ch; ++c) {
        ptr[j * ch + c] = ptr[i * ch + c];
      }
      j++; // update j
    }
  }

  // update node num
  points.mutable_info().set_pt_num(pt_num_remain);
}