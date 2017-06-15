#ifndef CAFFE_OCTREE_DATABASE_LAYER_HPP_
#define CAFFE_OCTREE_DATABASE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/octree.hpp"
#include "caffe/util/db.hpp"

namespace caffe
{
	template <typename Dtype>
	class OctreeDataBaseLayer : public Layer<Dtype>, public InternalThread 
	{
	public:
		explicit OctreeDataBaseLayer(const LayerParameter& param);
		virtual ~OctreeDataBaseLayer();

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		// Data layers have no bottoms, so reshaping is trivial.
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}

		// forward
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		// backward
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

		// layer type
		virtual inline const char* type() const { return "OctreeDataBase"; }

		// blob number
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:
		virtual void InternalThreadEntry();
		virtual void load_batch(OctreeBatch<Dtype>* batch);
		void Next();
		bool Skip();

	protected:
		// for prefetch thread
		vector<shared_ptr<OctreeBatch<Dtype> > > prefetch_;
		BlockingQueue<OctreeBatch<Dtype>*> prefetch_free_;
		BlockingQueue<OctreeBatch<Dtype>*> prefetch_full_;
		OctreeBatch<Dtype>* prefetch_current_;

		bool output_labels_; // true if #top > 1		
		bool segmentation_;  // true if used for segmentation
		int batch_size_;
		int curr_depth_;

		vector<vector<char> > data_buffer_;
		vector<int> label_buffer_;
		unsigned int rand_skip_;

		// data reader
		uint64_t offset_;
		shared_ptr<db::DB> db_;
		shared_ptr<db::Cursor> cursor_;
	};
}  // namespace caffe
#endif  // CAFFE_OCTREE_DATABASE_LAYER_HPP_
