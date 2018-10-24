#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	LossLayer<Dtype>::LayerSetUp(bottom, top);	
	
	if (this->layer_param_.loss_param().has_normalization()) {
		normalization_ = this->layer_param_.loss_param().normalization();
	}
	else {
		normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
	}
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->num_axes() > 2 ? bottom[0]->count(2) : 1;
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::get_normalizer(
	LossParameter_NormalizationMode normalization_mode) {
	Dtype normalizer;
	switch (normalization_mode) {
	case LossParameter_NormalizationMode_FULL:
	case LossParameter_NormalizationMode_VALID:
		normalizer = Dtype(outer_num_ * inner_num_);
		break;
	case LossParameter_NormalizationMode_BATCH_SIZE:
		normalizer = Dtype(outer_num_);
		break;
	case LossParameter_NormalizationMode_NONE:
		normalizer = Dtype(1);
		break;
	default:
		LOG(FATAL) << this->layer_param_.name() << " Unknown normalization mode: "
			<< LossParameter_NormalizationMode_Name(normalization_mode);
	}
	// Some users will have no labels for some examples in order to 'turn off' a
	// particular loss in a multi-task setup. The max prevents NaNs in that case.
	return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / get_normalizer(normalization_) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / get_normalizer(normalization_);
      caffe_cpu_axpby(
          bottom[i]->count(),					// count
          alpha,								// alpha
          diff_.cpu_data(),						// a
          Dtype(0),								// beta
          bottom[i]->mutable_cpu_diff());		// b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
