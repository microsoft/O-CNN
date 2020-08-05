#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "points_parser.h"

namespace tensorflow {

REGISTER_OP("PointsProperty")
    .Input("points: string")
    .Attr("property_name: string")
    .Attr("channel: int")
    .Output("out_property: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int channel;
      TF_RETURN_IF_ERROR(c->GetAttr("channel", &channel));
      c->set_output(0, c->MakeShape({c->UnknownDim(), channel}));
      return Status::OK();
    })
    .Doc(R"doc(Points property operator.)doc");

class PointsPropertyOp : public OpKernel {
 public:
  explicit PointsPropertyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("property_name", &property_name_));
    OP_REQUIRES_OK(context, context->GetAttr("channel", &channel_));
  }

  void Compute(OpKernelContext* context) override {
    // input points
    const Tensor& data_in = context->input(0);
    auto points_buffer = data_in.flat<string>();
    int batch_size = data_in.NumElements();
    vector<PointsParser> points_in(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      points_in[i].set(points_buffer(i).data());
    }

    // pt number
    int total_num = 0;
    vector<int> npts(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      npts[i] = points_in[i].info().pt_num();
      total_num += npts[i];
    }

    // channel
    PointsInfo::PropType ptype;
    if (property_name_ == "xyz") {
      ptype = PointsInfo::kPoint;
    } else if (property_name_ == "label") {
      ptype = PointsInfo::kLabel;
    } else if (property_name_ == "normal") {
      ptype = PointsInfo::kNormal;
    } else if (property_name_ == "feature") {
      ptype = PointsInfo::kFeature;
    } else {
      LOG(FATAL) << "Unsupported Property: " << property_name_;
    }
    int channel = channel_;  // `channel` is the actual channel saved in points
    if (ptype == PointsInfo::kPoint) {
      CHECK(channel_ == 3 || channel_ == 4)
          << "The specified channel_ for xyz is wrong.";
      channel = 3;
    }
    for (int i = 0; i < batch_size; ++i) {
      int ch = points_in[i].info().channel(ptype);
      CHECK_EQ(ch, channel) << "The specified channel_ is wrong.";
      const float* ptr = points_in[i].ptr(ptype);
      CHECK_NE(ptr, nullptr) << "The specified property does not exist.";
    }

    // init output
    Tensor* out_tensor;
    TensorShape out_shp({total_num, channel_});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shp, &out_tensor));
    float* out_ptr = out_tensor->flat<float>().data();

    // copy output
    for (int i = 0; i < batch_size; ++i) {
      const float* in_ptr = points_in[i].ptr(ptype);
      for (int j = 0; j < npts[i]; ++j) {
        int jxc_in = j * channel, jxc_out = j * channel_;
        for (int c = 0; c < channel; ++c) {
          out_ptr[jxc_out + c] = in_ptr[jxc_in + c];
        }
      }
      out_ptr += npts[i] * channel_;
    }

    // output point index if channel == 4
    if (ptype == PointsInfo::kPoint && channel_ == 4) {
      out_ptr = out_tensor->flat<float>().data();
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < npts[i]; ++j) {
          out_ptr[j * channel_ + 3] = static_cast<float>(i);  // id
        }
        out_ptr += npts[i] * channel_;
      }
    }
  }

 private:
  string property_name_;
  int channel_;
};

REGISTER_KERNEL_BUILDER(Name("PointsProperty").Device(DEVICE_CPU), PointsPropertyOp);

}  // namespace tensorflow
