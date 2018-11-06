#include "caffe/layers/octree_database_layer.hpp"
#include "caffe/layers/octree_property_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/octree.hpp"

namespace caffe {

template <typename Dtype>
OctreeDataBaseLayer<Dtype>::OctreeDataBaseLayer(const LayerParameter& param)
  : DataLayer<Dtype>(param) {}

template <typename Dtype>
OctreeDataBaseLayer<Dtype>::~OctreeDataBaseLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void OctreeDataBaseLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // get parameters
  CHECK(this->layer_param_.has_octree_param()) << "The octree_param must be set";
  signal_channel_ = this->layer_param_.octree_param().signal_channel();
  curr_depth_ = this->layer_param_.octree_param().curr_depth();

  rand_skip_ = this->layer_param_.data_param().rand_skip();
  batch_size_ = this->layer_param_.data_param().batch_size();
  CHECK_GT(batch_size_, 0) << "Positive batch size required";
  Octree::set_batchsize(batch_size_);
  octree_buffer_.resize(batch_size_);

  output_octree_ = top.size() == 3;

  // set the feature_layer_
  LayerParameter feature_param(this->layer_param_);
  feature_param.set_type("OctreeProperty");
  feature_param.mutable_octree_param()->set_content_flag("feature");
  feature_layer_ = LayerRegistry<Dtype>::CreateLayer(feature_param);
  feature_btm_vec_.resize(1);
  feature_top_vec_.resize(1);
  feature_top_vec_[0] = top[0];
  feature_layer_->SetUp(feature_btm_vec_, feature_top_vec_);

  // initialize top blob shape
  // a workaround for a valid first-time reshape
  vector<int> data_shape{ 1, signal_channel_, 8, 1 };
  top[0]->Reshape(data_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(data_shape);
  }
  if (this->output_labels_) {
    vector<int> label_shape{ batch_size_ };
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
  if (output_octree_) {
    top[2]->Reshape(vector<int> {1});
  }

  //// whether this layer needs dropout
  //int dropout_size = octree_param.dropout_depth_size();
  //CHECK_EQ(dropout_size, octree_param.dropout_ratio_size());
  //dropout_ = dropout_size != 0;
  //if (dropout_) {
  //  dropout_ratio_.resize(dropout_size);
  //  dropout_depth_.resize(dropout_size);
  //  for (int i = 0; i < dropout_size; ++i) {
  //    dropout_ratio_[i] = octree_param.dropout_ratio(i);
  //    dropout_depth_[i] = octree_param.dropout_depth(i);
  //  }
  //}
}

template<typename Dtype>
void OctreeDataBaseLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();

  Datum datum;
  Dtype* label_data = nullptr;
  if (this->output_labels_) label_data = batch->label_.mutable_cpu_data();
  for (int i = 0; i < batch_size_; ++i) {
    // get a datum
    while (this->Skip()) this->Next();
    datum.ParseFromString(this->cursor_->value());

    //// dropout octree nodes
    //if (dropout_) {
    //  if (signal_location_ == 0) {
    //    octree::octree_dropout(data_buffer_[i], datum.data(), dropout_depth_[0],
    //        dropout_ratio_[0], signal_channel_);
    //  } else {
    //    int drop_depth = RandDropDepth();
    //    octree::aoctree_dropout(data_buffer_[i], datum.data(), drop_depth, signal_channel_);
    //  }
    //}

    // copy data
    int n = datum.data().size();
    octree_buffer_[i].resize(n);
    memcpy(octree_buffer_[i].data(), datum.data().data(), n);
    if (this->output_labels_) label_data[i] = static_cast<Dtype>(datum.label());

    // update cursor
    this->Next();
  }

  // merge octrees
  octree::merge_octrees<Dtype>(batch->data_, octree_buffer_);

  //// rand skip a datum
  //static uint32 indicator = 1;
  //if (rand_skip_ > 0 && phase_ == TRAIN) {
  //  if (0 == (indicator % rand_skip_)) {
  //    int r = 0;
  //    caffe_rng_bernoulli(1, Dtype(0.5), &r);
  //    if (r > 0) {
  //      Datum& datum = *(reader_.full().pop("Waiting for data"));
  //      reader_.free().push(const_cast<Datum*>(&datum));
  //    }
  //  }
  //  indicator++;
  //}

  batch_timer.Stop();
  LOG_EVERY_N(INFO, 50) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

template<typename Dtype>
int OctreeDataBaseLayer<Dtype>::RandDropDepth() {
  int n = dropout_ratio_.size();
  vector<float> ratio_cum(n + 1, 0);
  for (int i = 0; i < n; ++i) {
    ratio_cum[i + 1] = ratio_cum[i] + dropout_ratio_[i];
  }
  float rnd = 0.0f;
  caffe_rng_uniform<float>(1, 0.0f, 1.0f, &rnd);
  for (int i = 0; i < n; ++i) {
    if (rnd < ratio_cum[i + 1]) return dropout_depth_[i];
  }
  return 20;
}

template <typename Dtype>
void OctreeDataBaseLayer<Dtype>::Forward_cpu(
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
    top[1]->set_cpu_data(this->prefetch_current_->label_.mutable_cpu_data());
  }

  // set the global octree
  Blob<Dtype>& the_octree = Octree::get_octree(Dtype(0));
  the_octree.ReshapeLike(curr_octree);
  the_octree.set_cpu_data(curr_octree.mutable_cpu_data());

  // set octree - top[2]
  if (output_octree_) {
    top[2]->ReshapeLike(curr_octree);
    top[2]->set_cpu_data(curr_octree.mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(OctreeDataBaseLayer, Forward);
#endif

INSTANTIATE_CLASS(OctreeDataBaseLayer);
REGISTER_LAYER_CLASS(OctreeDataBase);

}  // namespace caffe
