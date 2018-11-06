#include "caffe/layers/octree_database_layer.hpp"
#include "caffe/util/octree.hpp"

namespace caffe {

template <typename Dtype>
void OctreeDataBaseLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->prefetch_current_) this->prefetch_free_.push(this->prefetch_current_);
  this->prefetch_current_ = this->prefetch_full_.pop("Waiting for data");
  Blob<Dtype>& curr_octree = this->prefetch_current_->data_;

  // set data - top[0]
  feature_btm_vec_[0] = &curr_octree;
  feature_layer_->Reshape(feature_btm_vec_, feature_top_vec_);
  feature_layer_->Forward(feature_btm_vec_, feature_top_vec_);

  // set label - top[1]
  if (this->output_labels_) {
    top[1]->ReshapeLike(this->prefetch_current_->label_);
    top[1]->set_gpu_data(this->prefetch_current_->label_.mutable_gpu_data());
  }

  // set the global octree
  Blob<Dtype>& the_octree = Octree::get_octree(Dtype(0));
  the_octree.ReshapeLike(curr_octree);
  the_octree.set_gpu_data(curr_octree.mutable_gpu_data());

  // set octree - top[2]
  if (output_octree_) {
    top[2]->ReshapeLike(curr_octree);
    top[2]->set_gpu_data(curr_octree.mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(OctreeDataBaseLayer);

}  // namespace caffe
