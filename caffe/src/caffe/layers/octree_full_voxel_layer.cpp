#include <vector>

#include "caffe/layers/octree_full_voxel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{
	template<typename Dtype>
	void Octree2FullVoxelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{		
		curr_depth_ = Octree::get_curr_depth();
		batch_size_ = bottom[0]->shape(0);
		//CHECK_EQ(curr_depth_, this->layer_param_.octree_param().curr_depth());
	}

	template <typename Dtype>
	void Octree2FullVoxelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top)
	{		
		vector<int> top_shape(5);
		top_shape[0] = batch_size_;
		top_shape[1] = bottom[0]->shape(1);
		top_shape[2] = top_shape[3] = top_shape[4] = 1 << curr_depth_;

		if (top[0]->count() != 0)
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
			
			// batch size
			CHECK_EQ(batch_size_, *octree_batch_.batch_size_);

			// check full_octree_
			int full_layer_depth = *octree_batch_.full_layer_;
			CHECK_GE(full_layer_depth, curr_depth_);

			// check bottom height
			int bottom_h = bottom[0]->shape(2);
			CHECK_EQ(bottom_h, octree_batch_.node_num(curr_depth_))
				<< "The OctreePaddingLayer should be added before "
				<< this->layer_param_.name();
		}

		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void Octree2FullVoxelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{	
		// calc mapping
		if (index_mapper_.count() == 0) build_mapping(curr_depth_);

		int voxel_num = 1 << 3 * curr_depth_;
		int channel = bottom[0]->shape(1);
		int bottom_h = bottom[0]->shape(2);
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const unsigned* xyz_to_key = index_mapper_.cpu_data();
		for (int n = 0; n < batch_size_; ++n)
		{
			for (int c = 0; c < channel; ++c)
			{
				for (int k = 0; k < voxel_num; ++k)
				{
					top_data[(n*channel + c)*voxel_num + k] = 
						bottom_data[c*bottom_h + n*voxel_num + xyz_to_key[k]];
				}
			}
		}
	}
	
	template <typename Dtype>
	void Octree2FullVoxelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
	{
		if (!propagate_down[0]) { return; }

		int voxel_num = 1 << 3 * curr_depth_;
		int channel = bottom[0]->shape(1);
		int bottom_h = bottom[0]->shape(2);
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const unsigned* xyz_to_key = index_mapper_.cpu_data();
		for (int n = 0; n < batch_size_; ++n)
		{
			for (int c = 0; c < channel; ++c)
			{
				for (int i = 0; i < voxel_num; ++i)
				{
					bottom_diff[c*bottom_h + n*voxel_num + xyz_to_key[i]] =
						top_diff[(n*channel + c)*voxel_num + i];
				}
			}
		}
	}

	template <typename Dtype>
	void Octree2FullVoxelLayer<Dtype>::build_mapping(int depth)
	{
		int n = 1 << depth;
		vector<int> mapper_shape{ n*n*n };
		index_mapper_.Reshape(mapper_shape);
		unsigned * mapper_ptr = index_mapper_.mutable_cpu_data();
		for (unsigned x = 0; x < n; ++x )
		{
			for (unsigned y = 0; y < n; ++y)
			{
				for (unsigned z = 0; z < n; ++z)
				{
					// xyz index
					unsigned xyz = (n*x + y) * n + z;

					// key
					unsigned key = 0;
					for (int i = 0; i < depth; i++)
					{
						unsigned mask = 1u << i;
						key |= ((x & mask) << (2 * i + 2)) |
							((y & mask) << (2 * i + 1)) |
							((z & mask) << (2 * i));
					}

					// mapping
					mapper_ptr[xyz] = key;
				}
			}
		}
	}

	#ifdef CPU_ONLY
	STUB_GPU(Octree2FullVoxelLayer);
	#endif

	INSTANTIATE_CLASS(Octree2FullVoxelLayer);
	REGISTER_LAYER_CLASS(Octree2FullVoxel);
}  // namespace caffe
