#include "simplify_points.h"
#include "math_functions.h"


SimplifyPoints::SimplifyPoints(int dim, bool avg_points, float offset) {
  dim_ = dim;
  offset_ = offset;
  avg_points_ = avg_points;
  spatial_hash_.resize(dim_ * dim_ * dim_);
}

string SimplifyPoints::set_point_cloud(const string filename) {
  // load point cloud
  string error_msg;
  bool succ = point_cloud_.read_points(filename);
  if (!succ) {
    error_msg = "Can not load " + filename;
    return error_msg;
  }
  succ = point_cloud_.info().check_format(error_msg);
  if (!succ) {
    error_msg = filename + ": " + error_msg;
    return error_msg;
  }

  // deal with empty points
  int npt = point_cloud_.info().pt_num();
  if (npt == 0) {
    error_msg = filename + ": This is an empty points!";
    return error_msg;
  }
  return error_msg;
}

bool SimplifyPoints::write_point_cloud(const string filename) {
  return point_cloud_.write_points(filename);
}

void SimplifyPoints::simplify() {
  // init
  transform();
  spatial_hash_.assign(dim_ * dim_ * dim_, -1);
  vector<float> normal_output, pts_output;
  vector<int> pt_num;

  // average
  const float* pts = point_cloud_.ptr(PointsInfo::kPoint);
  const float* normals = point_cloud_.ptr(PointsInfo::kNormal);
  int nnum = point_cloud_.info().pt_num();
  for (int i = 0, id = 0; i < nnum; ++i) {
    // hash
    int ix3 = i * 3;
    int x = static_cast<int>(pts[ix3]);
    int y = static_cast<int>(pts[ix3 + 1]);
    int z = static_cast<int>(pts[ix3 + 2]);
    int h = (x * dim_ + y) * dim_ + z;

    if (spatial_hash_[h] == -1) {
      spatial_hash_[h] = id++;

      pt_num.push_back(0);
      for (int c = 0; c < 3; ++c) {
        pts_output.push_back(0.0f);
        normal_output.push_back(0.0f);
      }
    }

    int j = spatial_hash_[h];
    int jx3 = j * 3;
    pt_num[j] += 1;

    if (avg_points_) {
      for (int c = 0; c < 3; ++c) {
        normal_output[jx3 + c] += normals[ix3 + c];
        pts_output[jx3 + c] += pts[ix3 + c];
      }
    } else {
      for (int c = 0; c < 3; ++c) {
        normal_output[jx3 + c] = normals[ix3 + c];
        pts_output[jx3 + c] = pts[ix3 + c];
      }
    }
  }

  // normalize
  if (avg_points_) {
    int n = pt_num.size();
    for (int i = 0; i < n; ++i) {
      int ix3 = i * 3;
      float len = norm2(normal_output.data() + ix3, 3) + 1.0e-20f;
      for (int c = 0; c < 3; ++c) {
        normal_output[ix3 + c] /= len;
        pts_output[ix3 + c] /= pt_num[i];
      }
    }
  }

  point_cloud_.set_points(pts_output, normal_output);
  inv_transform();
}

void SimplifyPoints::transform() {
  // bounding sphere
  int npt = point_cloud_.info().pt_num();
  bounding_sphere(radius_, center_, point_cloud_.ptr(PointsInfo::kPoint), npt);
  
  // centralize & displacement
  radius_ *= 1.001f;
  if (offset_ > 1.0e-10f) {
    offest_obj_ = offset_ * 2.0f * radius_ / float(dim_);
    point_cloud_.displace(offest_obj_);
    radius_ += offest_obj_ * 1.001f;
  }
  float origin[3] = {
    radius_ - center_[0], radius_ - center_[1], radius_ - center_[2]
  };
  point_cloud_.translate(origin);  // translate the points to origin


  // scale
  float mul = dim_ / (2.0f * radius_);
  point_cloud_.uniform_scale(mul);
}

void SimplifyPoints::inv_transform() {
  // scale
  float mul = 2.0f * radius_ / dim_;
  point_cloud_.uniform_scale(mul);

  // translate & displacement
  float origin[3] = {
    center_[0] - radius_, center_[1] - radius_, center_[2] - radius_
  };
  point_cloud_.translate(origin);
  if (offset_ > 1.0e-10f) {
    point_cloud_.displace(-offest_obj_);
  }

}
