#ifndef CAFFE_UTIL_OCTREE_HPP_
#define CAFFE_UTIL_OCTREE_HPP_

#include <vector>
#include "caffe/blob.hpp"
using std::vector;

namespace caffe {

	// A singleton class to hold global & common stuff for octree.
	class Octree
	{
	public:
		// Thread local context for Octree. 
		static Octree& Get();
		// this function should be called only-once in the data_layer
		static void set_octree_depth(int depth) { Get().depth_ = depth; }
		static int get_curr_depth() { return Get().curr_depth_; }
		static void set_curr_depth(int depth) { Get().curr_depth_ = depth; }
		static Blob<int>& get_octree() { return Get().octree_; }
		static Blob<float>& get_workspace(float) { return Get().workspace_; }
		static Blob<double>& get_workspace(double) { return Get().workspaced_; }
		static vector<shared_ptr<Blob<int> > >& get_ni() { return Get().ni_; }
		static Blob<int>& get_parent_array() { return Get().parent_; }
		static Blob<int>& get_dis_array() { return Get().displacement_; }

	protected:
		void init_neigh_index();

	protected:
		int depth_;
		int curr_depth_;
		Blob<int> octree_;

		// used to get the neighbor information
		vector<shared_ptr<Blob<int> > > ni_;
		// used to calculate the neighbor information
		Blob<int> parent_;
		Blob<int> displacement_;
		
		// workspace is used as the temporary buffer of 
		// gemm in octree_base_conv to save memory.
		// TODO: limit the maximum size of workspace for very deep octrees.
		Blob<float> workspace_;
		Blob<double> workspaced_;

	private:
		// The private constructor to avoid duplicate instantiation.
		Octree() : depth_(0), curr_depth_(0), octree_(), ni_(4), parent_(),
			displacement_(), workspace_(), workspaced_() 
		{
			init_neigh_index();
		}
	};

namespace octree{
	template<typename Dtype>
	void pad_forward_cpu(Dtype* Y, const int Hy,
		const int Cy, const Dtype* X, const int Hx, const int* label);
	template<typename Dtype>
	void pad_forward_gpu(Dtype* Y, const int Hy,
		const int Cy, const Dtype* X, const int Hx, const int* label);
	template<typename Dtype>
	void pad_backward_cpu(Dtype* X, const int Hx,
		const int Cx, const Dtype* Y, const int Hy, const int* label);
	template<typename Dtype>
	void pad_backward_gpu(Dtype* X, const int Hx,
		const int Cx, const Dtype* Y, const int Hy, const int* label);

	template <typename Dtype>
	void octree2col_cpu(Dtype* data_col, const Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni);
	template <typename Dtype>
	void octree2col_gpu(Dtype* data_col, const Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni);

	template <typename Dtype>
	void col2octree_cpu(const Dtype* data_col, Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni);
	template <typename Dtype>
	void col2octree_gpu(const Dtype* data_col, Dtype* data_octree,
		const int channel, const int height, const int kernel_size,
		const int stride, const int* neigh, const int* ni);
	
	void calc_neigh_cpu(int* neigh_split, const int* neigh,
		const int* children, const int node_num);
	void calc_neigh_gpu(int* neigh_split, const int* neigh,
		const int* children, const int node_num);
	void calc_neigh_cpu(int* neigh, const int depth, const int batch_size);
	void calc_neigh_gpu(int* neigh, const int depth, const int batch_size);
	// calculate neighborhood information with the hash table
	void calc_neighbor(int* neigh, const unsigned* key, const int node_num,
		const int displacement);

	void generate_key_gpu(int* key_split, const int* key,
		const int* children, const int node_num);
	void generate_key_cpu(int* key_split, const int* key,
		const int* children, const int node_num);
	void generate_key_gpu(int* key, const int depth, const int batch_size);
	void generate_key_cpu(int* key, const int depth, const int batch_size);
	
	inline void compute_key(int& key, const int* pt, const int depth);
	inline void compute_pt(int* pt, const int& key, const int depth);
}

	class OctreeParser
	{
	public:
		OctreeParser(const void* data);

		int total_node_number() { return *total_node_num_; }
		int final_node_number() { return *final_node_num_; }
		// todo: modify octree formate, and remove this function
		int node_number_nempty(int depth); 

	public:
		// original pointer
		const void* metadata_;

		// octree header information
		const int* total_node_num_;
		const int* final_node_num_;
		const int* depth_;
		const int* full_layer_;
		const int* node_num_;
		const int* node_num_accu_;

		// octree structure
		const int* key_;
		const int* children_;

		// octree data
		const int* signal_;
		const int* seg_label_;
	};

	class OctreeBatchParser
	{
	public:

		// set pointers
		void set_cpu(void* data, int depth = 0, int total_nnum = 0,
			int batch_size = 0, int content_flags = 0);		
		void set_gpu(void* d_data) { d_metadata_ = (int*) d_data; }

		// return the total node number
		int total_num() { return node_num_cum_[(*depth_) + 1]; }

		// node number in the specified layer
		int node_num(int depth) { return node_num_[depth]; }
		
		// non-empty node number int the specified layer
		int node_num_nempty(int depth) { return node_num_nempty_[depth]; }

		// point to the first children of the specified layer
		int* children_cpu(int depth) { return children_ + node_num_cum_[depth]; }
		int* children_gpu(int depth) { return d_metadata_ + (children_cpu(depth) - metadata_); }

		// point to the first children of the specified layer
		int* key_cpu(int depth) { return key_ + node_num_cum_[depth]; }
		int* key_gpu(int depth) { return d_metadata_ + (key_cpu(depth) - metadata_); }

		// point to the first neighbor of the specified layer
		int* neighbor_cpu(int depth) { return neigh_ + AVG_NGH_NUM*node_num_cum_[depth]; }
		int* neighbor_gpu(int depth) { return d_metadata_ + (neighbor_cpu(depth) - metadata_); }

		// size of int
		static int header_sizeofint(const int depth, const int batch_size);
		static int octree_batch_sizeofint(const int depth, const int total_nnum,
			const int batch_size, int content_flags);

	public:
		// original pointer
		int* metadata_;
		int* d_metadata_;

		// octree header information
		int* batch_size_;
		int* depth_;
		int* full_layer_;
		int* node_num_;			// node number of each depth
		int* node_num_cum_;		// cumulative node number
		int* node_num_nempty_;	// non-empty node number of each depth
		int* node_num_oct_;		// node number of each depth layer
		int* node_num_nempty_oct_;
		int* content_flags_;	

		// octree structure information
		int* key_;
		int* children_;
		int* neigh_;

		// average neighbor number
		static const int AVG_NGH_NUM = 8;		
	};
	
	// TODO: inherit class OctreeBatch<Dtype> from the class Batch<Dtype>, so 
	// that we can remove the explicit 
	// instantiate of BlockingQueue<OctreeBatch<Dtype>*> in blocking_queue.cpp 
	// and reuse some code from the base_data_layer. 
	template <typename Dtype>
	class OctreeBatch
	{
	public:
		void set_octreebatch(const vector<vector<char> >& octree_buffer,
			const vector<int>& label_buffer, const bool segmentation = false);

	public:
		Blob<Dtype> data_, label_;
		Blob<int> octree_;
	};

}  // namespace caffe

#endif
