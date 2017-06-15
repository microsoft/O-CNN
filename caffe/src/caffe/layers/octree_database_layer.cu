#include <vector>

#include "caffe/layers/octree_database_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void OctreeDataBaseLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		if (prefetch_current_) prefetch_free_.push(prefetch_current_);
		prefetch_current_ = prefetch_full_.pop("Waiting for data");

		// set octree - top[0]
		top[0]->ReshapeLike(prefetch_current_->data_);
		top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());

		// set label - top[1]
		if (output_labels_)
		{
			top[1]->ReshapeLike(prefetch_current_->label_);
			top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
		}

		// set octree 
		Blob<int>& the_octree = Octree::get_octree();
		the_octree.ReshapeLike(prefetch_current_->octree_);
		the_octree.set_gpu_data(prefetch_current_->octree_.mutable_gpu_data());
	}

INSTANTIATE_LAYER_GPU_FORWARD(OctreeDataBaseLayer);
}  // namespace caffe
