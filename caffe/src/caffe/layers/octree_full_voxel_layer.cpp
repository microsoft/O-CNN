#include <vector>

#include "caffe/layers/octree_full_voxel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void Octree2FullVoxelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();
  //curr_depth_ = Octree::get_curr_depth();

  //batch_size_ = bottom[0]->shape(0);
  CHECK(this->layer_param_.octree_param().has_batch_size())
      << "Error in " << this->layer_param_.name() << ": "
      << "The batch size of input octree should be set coreectly.";
  batch_size_ = this->layer_param_.octree_param().batch_size();
}

template <typename Dtype>
void Octree2FullVoxelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(5);
  top_shape[0] = batch_size_;
  top_shape[1] = bottom[0]->shape(1);
  top_shape[2] = top_shape[3] = top_shape[4] = 1 << curr_depth_;

  if (top[0]->count() != 0) {
    bool octree_in = bottom.size() == 2;
    Blob<Dtype>& the_octree = octree_in ? *bottom[1] : Octree::get_octree(Dtype(0));
    octree::set_octree_parser(octree_batch_, the_octree);

    // batch size
    CHECK_EQ(batch_size_, octree_batch_.info().batch_size())
        << "The batch_size_ is wrong in the layer: " << this->layer_param_.name();

    // check full_octree_
    int full_layer_depth = octree_batch_.info().full_layer();
    CHECK_GE(full_layer_depth, curr_depth_)
        << "The current_depth_ is wrong in the layer: " << this->layer_param_.name();

    // check bottom height
    int bottom_h = bottom[0]->shape(2);
    CHECK_EQ(bottom_h, octree_batch_.info().node_num(curr_depth_))
        << "The node number is wrong in the layer: " << this->layer_param_.name();
  }

  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void Octree2FullVoxelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // calc mapping
  if (index_mapper_.count() == 0) build_mapping(curr_depth_);

  int voxel_num = 1 << 3 * curr_depth_;
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const unsigned* xyz_to_key = index_mapper_.cpu_data();
  for (int n = 0; n < batch_size_; ++n) {
    for (int c = 0; c < channel; ++c) {
      for (int k = 0; k < voxel_num; ++k) {
        top_data[(n * channel + c) * voxel_num + k] =
            bottom_data[c * bottom_h + n * voxel_num + xyz_to_key[k]];
      }
    }
  }
}

template <typename Dtype>
void Octree2FullVoxelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int voxel_num = 1 << 3 * curr_depth_;
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const unsigned* xyz_to_key = index_mapper_.cpu_data();
  for (int n = 0; n < batch_size_; ++n) {
    for (int c = 0; c < channel; ++c) {
      for (int i = 0; i < voxel_num; ++i) {
        bottom_diff[c * bottom_h + n * voxel_num + xyz_to_key[i]] =
            top_diff[(n * channel + c) * voxel_num + i];
      }
    }
  }
}

template <typename Dtype>
void Octree2FullVoxelLayer<Dtype>::build_mapping(int depth) {
  int n = 1 << depth;
  vector<int> mapper_shape{ n*n * n };
  index_mapper_.Reshape(mapper_shape);
  unsigned * mapper_ptr = index_mapper_.mutable_cpu_data();
  for (unsigned x = 0; x < n; ++x) {
    for (unsigned y = 0; y < n; ++y) {
      for (unsigned z = 0; z < n; ++z) {
        // xyz index
        unsigned xyz = (n * x + y) * n + z;

        // key
        unsigned key = 0;
        for (int i = 0; i < depth; i++) {
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
