#include <boost/thread.hpp>
#include <vector>

#include "caffe/layers/octree_database_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {
	template <typename Dtype>
	OctreeDataBaseLayer<Dtype>::OctreeDataBaseLayer(const LayerParameter& param)
		: Layer<Dtype>(param), prefetch_(param.data_param().prefetch()),
		prefetch_free_(), prefetch_full_(), prefetch_current_(), offset_()
	{
		for (int i = 0; i < prefetch_.size(); ++i) 
		{
			prefetch_[i].reset(new OctreeBatch<Dtype>());
			prefetch_free_.push(prefetch_[i].get());
		}

		db_.reset(db::GetDB(param.data_param().backend()));
		db_->Open(param.data_param().source(), db::READ);
		cursor_.reset(db_->NewCursor());
	}
	
	template <typename Dtype>
	OctreeDataBaseLayer<Dtype>::~OctreeDataBaseLayer()
	{
		this->StopInternalThread();
	}
	
	template <typename Dtype>
	void OctreeDataBaseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		// true if #top > 1
		output_labels_ = top.size() == 2;
		
		// batch size
		batch_size_ = this->layer_param_.data_param().batch_size();
		CHECK_GT(batch_size_, 0) << "Positive batch size required";

		rand_skip_ = this->layer_param_.data_param().rand_skip();
		// whether this layer is for segmentation
		segmentation_ = this->layer_param_.octree_param().segmentation();

		// buffer for the dutam from data base
		data_buffer_.resize(batch_size_);
		label_buffer_.resize(batch_size_);
		
		// Read a data point, and use it to initialize the top blob.
		//Datum datum;
		//datum.ParseFromString(cursor_->value());
		//OctreeParser octree_parser(datum.data().data());
		//int octree_depth = *octree_parser.depth_;
		//Octree::set_octree_depth(octree_depth);
		//Octree::set_curr_depth(octree_depth);
		// int node_num = octree_parser.node_num_[octree_depth];

		// initialize top blob shape.
		// a workaround for a valid first-time reshape
		vector<int> data_shape, label_shape, octree_shape{ 1 };
		if (!segmentation_)
		{
			// note: this is just an estimation of the height of top blob
			// node_num * 1.2 : allocate slightly more memory to avoid re-allocating
			// int height = node_num * 1.2; 
			data_shape = { batch_size_, 3, 8, 1 };
			label_shape = { batch_size_ };
		}
		else{
			data_shape = { 1, 3, 8, 1 };
			label_shape = { 1, 8 };
		}			
		top[0]->Reshape(data_shape);
		for (int i = 0; i < prefetch_.size(); ++i)
		{
			prefetch_[i]->data_.Reshape(data_shape);
			prefetch_[i]->octree_.Reshape(octree_shape);
		}
		if (output_labels_)
		{
			top[1]->Reshape(label_shape);
			for (int i = 0; i < prefetch_.size(); ++i)
			{
				prefetch_[i]->label_.Reshape(label_shape);
			}
		}

		// Before starting the prefetch thread, we make cpu_data and gpu_data
		// calls so that the prefetch thread does not accidentally make simultaneous
		// cudaMalloc calls when the main thread is running. In some GPUs this
		// seems to cause failures if we do not so.
		for (int i = 0; i < prefetch_.size(); ++i)
		{
			prefetch_[i]->data_.mutable_cpu_data();
			prefetch_[i]->octree_.mutable_cpu_data();
			if (this->output_labels_) 
			{
				prefetch_[i]->label_.mutable_cpu_data();
			}
		}
		#ifndef CPU_ONLY
		if (Caffe::mode() == Caffe::GPU) 
		{
			for (int i = 0; i < prefetch_.size(); ++i) 
			{
				prefetch_[i]->data_.mutable_gpu_data();
				prefetch_[i]->octree_.mutable_gpu_data();
				if (this->output_labels_) 
				{
					prefetch_[i]->label_.mutable_gpu_data();
				}
			}
		}
		#endif

		// start internal thread for reading data
		DLOG(INFO) << "Initializing prefetch";
		StartInternalThread();
		DLOG(INFO) << "Prefetch initialized.";
	}

	template <typename Dtype>
	bool OctreeDataBaseLayer<Dtype>::Skip()
	{
		int size = Caffe::solver_count();
		int rank = Caffe::solver_rank();
		bool keep = (offset_ % size) == rank ||
			// In test mode, only rank 0 runs, so avoid skipping
			this->layer_param_.phase() == TEST;
		return !keep;
	}

	template<typename Dtype>
	void OctreeDataBaseLayer<Dtype>::Next() 
	{
		cursor_->Next();
		if (!cursor_->valid())
		{
			LOG_IF(INFO, Caffe::root_solver())
				<< "Restarting data prefetching from start.";
			cursor_->SeekToFirst();
		}
		offset_++;
	}


	template <typename Dtype>
	void OctreeDataBaseLayer<Dtype>::InternalThreadEntry()
	{
		#ifndef CPU_ONLY
		cudaStream_t stream;
		if (Caffe::mode() == Caffe::GPU)
		{
			CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		}
		#endif

		try
		{
			while (!must_stop())
			{
				OctreeBatch<Dtype>* batch = prefetch_free_.pop();
				load_batch(batch);

				#ifndef CPU_ONLY
				if (Caffe::mode() == Caffe::GPU)
				{
					batch->data_.data().get()->async_gpu_push(stream);
					batch->octree_.data().get()->async_gpu_push(stream);
					if(output_labels_) batch->label_.data().get()->async_gpu_push(stream);
					CUDA_CHECK(cudaStreamSynchronize(stream));
				}
				#endif

				prefetch_full_.push(batch);
			}
		}
		catch (boost::thread_interrupted&)
		{
			// Interrupted exception is expected on shutdown
		}

		#ifndef CPU_ONLY
		if (Caffe::mode() == Caffe::GPU) {
			CUDA_CHECK(cudaStreamDestroy(stream));
		}
		#endif
	}
	
	// This function is called on prefetch thread
	template<typename Dtype>
	void OctreeDataBaseLayer<Dtype>::load_batch(OctreeBatch<Dtype>* batch)
	{
		CPUTimer batch_timer;
		batch_timer.Start();

		//// rand skip one point
		//static uint32 indicator = 1;
		//if (rand_skip_ > 0 && phase_ == TRAIN)
		//{
		//	if (0 == (indicator % rand_skip_))
		//	{
		//		int r = 0;
		//		caffe_rng_bernoulli(1, Dtype(0.5), &r);
		//		if (r > 0)
		//		{
		//			Datum& datum = *(reader_.full().pop("Waiting for data"));
		//			reader_.free().push(const_cast<Datum*>(&datum));
		//		}				
		//	}
		//	indicator++;
		//}

		// get data from data reader
		Datum datum;
		for (int i = 0; i < batch_size_; ++i)
		{
			// get a datum
			while (Skip()) Next();
			datum.ParseFromString(cursor_->value());

			// copy data
			int n = datum.data().size();
			data_buffer_[i].resize(n);
			caffe_copy(n, datum.data().data(), data_buffer_[i].data());

			// Copy label
			if (this->output_labels_) label_buffer_[i] = datum.label();

			// update cursor
			Next();
		}
		
		// set batch
		batch->set_octreebatch(data_buffer_, label_buffer_, segmentation_);

		batch_timer.Stop();
		LOG_EVERY_N(INFO, 50) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	}

	template <typename Dtype>
	void OctreeDataBaseLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		if (prefetch_current_) prefetch_free_.push(prefetch_current_);
		prefetch_current_ = prefetch_full_.pop("Waiting for data");

		// set octree - top[0]
		top[0]->ReshapeLike(prefetch_current_->data_);
		top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());

		// set label - top[1]
		if (output_labels_)
		{
			top[1]->ReshapeLike(prefetch_current_->label_);
			top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
		}

		// set octree 
		Blob<int>& the_octree = Octree::get_octree();
		the_octree.ReshapeLike(prefetch_current_->octree_);
		the_octree.set_cpu_data(prefetch_current_->octree_.mutable_cpu_data());
	}

#ifdef CPU_ONLY
	STUB_GPU_FORWARD(OctreeDataBaseLayer, Forward);
#endif

	INSTANTIATE_CLASS(OctreeDataBaseLayer);
	REGISTER_LAYER_CLASS(OctreeDataBase);

}  // namespace caffe
