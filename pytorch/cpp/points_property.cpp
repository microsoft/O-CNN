#include <octree/octree_nn.h>
#include <octree/points.h>
#include <octree/points_parser.h>

#include "ocnn.h"

namespace {

PointsInfo::PropType get_ptype(const string property) {
  PointsInfo::PropType ptype = PointsInfo::kPoint;
  if (property == "xyz" || property == "xyzi") {
    ptype = PointsInfo::kPoint;
  } else if (property == "label") {
    ptype = PointsInfo::kLabel;
  } else if (property == "normal") {
    ptype = PointsInfo::kNormal;
  } else if (property == "feature") {
    ptype = PointsInfo::kFeature;
  } else {
    LOG(FATAL) << "Unsupported Property: " << property;
  }
  return ptype;
}

vector<float> tensor2vector(const Tensor& data_in) {
  vector<float> vec;
  Tensor data = data_in.contiguous();  // !!! make sure the Tensor is contiguous
  const int64_t num = data.numel();
  if (num > 0) {
    const float* ptr = data.data_ptr<float>();
    vec.assign(ptr, ptr + num);
  }
  return vec;
}

}  // anonymous namespace

Tensor points_new(Tensor pts, Tensor normals, Tensor features, Tensor labels) {
  // input
  vector<float> pts_in = tensor2vector(pts);
  vector<float> normals_in = tensor2vector(normals);
  vector<float> features_in = tensor2vector(features);
  vector<float> labels_in = tensor2vector(labels);

  // create the point cloud
  Points point_cloud;
  bool succ = point_cloud.set_points(pts_in, normals_in, features_in, labels_in);
  CHECK(succ) << "Error occurs when setting points";
  const vector<char>& points_buf = point_cloud.get_buffer();

  // output
  size_t sz = points_buf.size();
  Tensor output = torch::zeros({(int64_t)sz}, torch::dtype(torch::kUInt8));
  memcpy(output.data_ptr<uint8_t>(), points_buf.data(), sz);
  return output;
}

Tensor points_set_property(Tensor points_in, Tensor data, string property) {
  CHECK_EQ(data.dim(), 2);
  int num = data.size(0);
  int channel = data.size(1);
  data = data.contiguous();  // !!! make sure the Tensor is contiguous

  // init the points
  Points pts;
  auto points_ptr = points_in.data_ptr<uint8_t>();
  vector<char> points_buf(points_ptr, points_ptr + points_in.numel());
  pts.set(points_buf.data());

  // check the points
  string msg;
  bool succ = pts.info().check_format(msg);
  CHECK(succ) << msg;

  // get prop
  PointsInfo::PropType ptype = get_ptype(property);

  // set the property
  float* ptr = pts.mutable_ptr(ptype);
  CHECK(ptr != nullptr) << "The property does not exist.";
  CHECK_EQ(channel, pts.info().channel(ptype)) << "The channel is wrong.";
  CHECK_EQ(num, pts.info().pt_num()) << "The point number is wrong.";
  memcpy_cpu(num * channel, data.data_ptr<float>(), ptr);

  // output
  size_t sz = points_buf.size();
  Tensor output = torch::zeros({(int64_t)sz}, torch::dtype(torch::kUInt8));
  memcpy(output.data_ptr<uint8_t>(), points_buf.data(), sz);
  return output;
}

Tensor points_property(Tensor points, string property) {
  PointsParser points_in;
  points_in.set(points.data_ptr<uint8_t>());
  PointsInfo::PropType ptype = get_ptype(property);

  int npt = points_in.info().pt_num();
  int channel = points_in.info().channel(ptype);
  const float* ptr = points_in.ptr(ptype);
  // CHECK_NE(ptr, nullptr) << "The specified property does not exist.";
  if (ptr == nullptr) {
    return Tensor();
  }

  Tensor data_out = torch::zeros({npt, channel}, torch::dtype(torch::kFloat32));
  float* out_ptr = data_out.data_ptr<float>();
  memcpy_cpu(channel * npt, ptr, out_ptr);
  return data_out;
}

Tensor points_batch_property(vector<Tensor> tensors, string property) {
  int batch_size = tensors.size();
  vector<PointsParser> points_in(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    points_in[i].set(tensors[i].data_ptr<uint8_t>());
  }

  // pt number
  int total_num = 0;
  vector<int> npts(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    npts[i] = points_in[i].info().pt_num();
    total_num += npts[i];
  }

  // check
  PointsInfo::PropType ptype = get_ptype(property);
  int channel = points_in[0].info().channel(ptype);
  for (int i = 0; i < batch_size; ++i) {
    int ch = points_in[i].info().channel(ptype);
    CHECK_EQ(ch, channel) << "The specified channel_out is wrong.";
    const float* ptr = points_in[i].ptr(ptype);
    CHECK_NE(ptr, nullptr) << "The specified property does not exist.";
  }
  int channel_out = channel;
  if (property == "xyzi") { channel_out = 4; }

  // init output
  Tensor output = torch::zeros({total_num, channel_out}, torch::dtype(torch::kFloat32));
  float* out_ptr = output.data_ptr<float>();

  // copy output
  for (int i = 0; i < batch_size; ++i) {
    const float* in_ptr = points_in[i].ptr(ptype);
    for (int j = 0; j < npts[i]; ++j) {
      int jxc_in = j * channel, jxc_out = j * channel_out;
      for (int c = 0; c < channel; ++c) {
        out_ptr[jxc_out + c] = in_ptr[jxc_in + c];
      }
    }
    out_ptr += npts[i] * channel_out;
  }

  // output point index if channel == 4
  if (property == "xyzi") {
    out_ptr = output.data_ptr<float>();
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < npts[i]; ++j) {
        out_ptr[j * channel_out + 3] = static_cast<float>(i);  // id
      }
      out_ptr += npts[i] * channel_out;
    }
  }

  return output;
}
