#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_binary_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void AccuracyBinaryLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(bottom[0]->count(), bottom[1]->count())
			<< "Error in " << this->layer_param().name();
		top[0]->Reshape(vector<int> { 4 });
	}

	template <typename Dtype>
	void AccuracyBinaryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_label = bottom[1]->cpu_data();

		int accuarcy[4] = { 0, 0, 0, 0 };
		int count = bottom[0]->count();
		for (int i = 0; i < count; ++i)
		{
			int label_gt = static_cast<int>(bottom_label[i]);
			int label = static_cast<int>(bottom_data[i] > 0);

			int ind = label_gt * 2 + label;
			accuarcy[ind] ++;
		}

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int i = 0; i < 4; ++i)
		{
			top_data[i] = Dtype(accuarcy[i]) / Dtype(count);
		}
	}

	template <typename Dtype>
	void AccuracyBinaryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		for (int i = 0; i < propagate_down.size(); ++i)
		{
			if (propagate_down[i]) NOT_IMPLEMENTED;
		}
	}

	INSTANTIATE_CLASS(AccuracyBinaryLayer);
	REGISTER_LAYER_CLASS(AccuracyBinary);

}  // namespace caffe
