#include <vector>
#include "caffe/layers/octree_padding_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{
	template<typename Dtype>
	void OctreePaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top)
	{		
		curr_depth_ = Octree::get_curr_depth();
		//CHECK_EQ(curr_depth_, this->layer_param_.octree_param().curr_depth());
	}

	template <typename Dtype>
	void OctreePaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top)
	{
		if (top[0]->count() == 0)
		{
			// a workaround for the first time reshape
			vector<int> top_shape = bottom[0]->shape();
			top_shape[2] = 8;
			top[0]->Reshape(top_shape);
		}
		else
		{
			// todo: find more elegent solution to avoid the use of const_cast
			Blob<int>& the_octree = Octree::get_octree();
			octree_batch_.set_cpu(const_cast<int*>(the_octree.cpu_data()));
			#ifndef CPU_ONLY
			if (Caffe::mode() == Caffe::GPU)
			{
				octree_batch_.set_gpu(const_cast<int*>(the_octree.gpu_data()));
			}
			#endif
			
			// reshape data_buffer_
			int bottom_h = bottom[0]->shape(2);
			int top_h = octree_batch_.node_num(curr_depth_);
			if (top_h == bottom_h)
			{
				LOG(INFO) << "The layer " << this->layer_param_.name() << "is redundant.";
			}
			else {
				CHECK_EQ(bottom_h, octree_batch_.node_num_nempty(curr_depth_));
			}

			vector<int>top_shape = bottom[0]->shape();
			top_shape[2] = top_h;
			top[0]->Reshape(top_shape);
		}

		if (top.size() == 2) top[1]->ReshapeLike(*bottom[1]);
	}

	template <typename Dtype>
	void OctreePaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{	

		// padding
		int channel = top[0]->shape(1);
		int top_h = top[0]->shape(2);
		int bottom_h = bottom[0]->shape(2);
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		if (top_h != bottom_h)
		{
			octree::pad_forward_cpu<Dtype>(top_data, top_h, channel,
				bottom_data, bottom_h, octree_batch_.children_cpu(curr_depth_));
		}
		else{
			caffe_copy(bottom[0]->count(), bottom_data, top_data);
		}		
	}	

	template <typename Dtype>
	void OctreePaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
	{
		if (!propagate_down[0]) { return; }

		// padding
		int channel = top[0]->shape(1);
		int top_h = top[0]->shape(2);
		int bottom_h = bottom[0]->shape(2);
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* top_diff = top[0]->cpu_diff();
		if (top_h != bottom_h)
		{
			octree::pad_backward_cpu<Dtype>(bottom_diff, bottom_h, channel, 
				top_diff, top_h, octree_batch_.children_cpu(curr_depth_));
		}
		else{
			caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
		}
	}
	
	#ifdef CPU_ONLY
	STUB_GPU(OctreePaddingLayer);
	#endif

	INSTANTIATE_CLASS(OctreePaddingLayer);
	REGISTER_LAYER_CLASS(OctreePadding);
}  // namespace caffe
