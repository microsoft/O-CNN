#include "points_parser.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("PointsProperty")
    .Input("points: string")
    .Attr("property_name: string")
    .Attr("channel: int")
    .Output("out_property: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int channel;
      TF_RETURN_IF_ERROR(c->GetAttr("channel", &channel));
      c->set_output(0, c->MakeShape({ c->UnknownDim(), channel }));
      return Status::OK();
    })
    .Doc(R"doc(Points property operator.)doc");

class PointsPropertyOp : public OpKernel {
 public:
  explicit PointsPropertyOp(OpKernelConstruction* context) :
    OpKernel(context) {
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
    int channel = 0;
    if (property_name_ == "xyz") {
      channel = 4; // x, y, z, id
    } else if (property_name_ == "label") {
      channel = 1;
    } else {
      LOG(FATAL) << "Unsupported Property: " << property_name_;
    }
    CHECK_EQ(channel_, channel) << "The specified channel_ is wrong.";

    // init output
    Tensor* out_tensor;
    TensorShape out_shape({ total_num, channel });
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out_tensor));
    float* out_ptr = out_tensor->flat<float>().data();

    // copy output
    if (property_name_ == "xyz") {
      for (int i = 0; i < batch_size; ++i) {
        const float *in_ptr = points_in[i].points();
        for (int j = 0; j < npts[i]; ++j) {
          int jx3 = j * 3, jx4 = j * 4;
          for (int c = 0; c < 3; ++c) {
            out_ptr[jx4 + c] = in_ptr[jx3 + c];
          }
          out_ptr[jx4 + 3] = static_cast<float>(i); // id
        }
        out_ptr += npts[i] * channel;
      }
    } else if (property_name_ == "label") {
      for (int i = 0; i < batch_size; ++i) {
        const float *in_ptr = points_in[i].label();
        CHECK(in_ptr != nullptr) << "The points have on labels";
        for (int j = 0; j < npts[i]; ++j) {
          out_ptr[j] = in_ptr[j];
        }
        out_ptr += npts[i];
      }
    } else {
      LOG(FATAL) << "Unsupported Property: " << property_name_;
    }
  }

 private:
  string property_name_;
  int channel_;
};

REGISTER_KERNEL_BUILDER(Name("PointsProperty").Device(DEVICE_CPU), PointsPropertyOp);

}  // namespace tensorflow
