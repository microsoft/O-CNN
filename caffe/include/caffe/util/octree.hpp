#ifndef CAFFE_UTIL_OCTREE_HPP_
#define CAFFE_UTIL_OCTREE_HPP_

#include <vector>
#include <unordered_map>
#include "caffe/blob.hpp"
#include "caffe/util/octree_info.hpp"
#include "caffe/util/octree_parser.hpp"

using std::vector;
using std::unordered_map;

namespace caffe {

// A singleton class to hold global & common stuff for octree.
class Octree {
 public:
  // Thread local context for Octree.
  static Octree& Get();
  // this function should be called only-once in the data_layer
  static void set_octree_depth(int depth) { Get().depth_ = depth; }
  static int get_curr_depth() { return Get().curr_depth_; }
  static void set_curr_depth(int depth) { Get().curr_depth_ = depth; }
  static void set_batchsize(int bs) { Get().batch_size_ = bs; }
  static int get_batchsize() { return Get().batch_size_; }
  static int get_workspace_maxsize() { return Get().workspace_sz_; }
  static void set_workspace_maxsize(int sz) { Get().workspace_sz_ = sz; }
  static Blob<int>& get_parent_array() { return Get().parent_; }
  static Blob<int>& get_dis_array() { return Get().displacement_; }
  static Blob<float>& get_octree(float) { return Get().octree_; }
  static Blob<double>& get_octree(double) { return Get().octreed_; }
  static shared_ptr<Blob<float> > get_workspace(float, int id = 0);
  static shared_ptr<Blob<double> > get_workspace(double, int id = 0);
  static shared_ptr<Blob<int> > get_ni(const vector<int>& kernel_size);

 protected:
  void init_neigh_index();

 protected:
  int depth_;
  int curr_depth_;
  int batch_size_;
  int workspace_sz_;
  Blob<float> octree_;
  Blob<double> octreed_;

  // used to get the neighbor information
  vector<shared_ptr<Blob<int> > > ni_;
  unordered_map<string, int> ni_map_;

  // used to calculate the neighbor information
  Blob<int> parent_;
  Blob<int> displacement_;

  // workspace is used as the temporary buffer of
  // gemm in octree_base_conv to save memory.
  vector<shared_ptr<Blob<float> > > workspace_;
  vector<shared_ptr<Blob<double> > >  workspaced_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Octree() : depth_(0), curr_depth_(0), batch_size_(1), workspace_sz_(256 * 1024 * 1024),
    octree_(), parent_(), displacement_(), workspace_(), workspaced_() { init_neigh_index(); }
};

namespace octree {

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
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n);
template <typename Dtype>
void octree2col_gpu(Dtype* data_col, const Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n);

template <typename Dtype>
void col2octree_cpu(const Dtype* data_col, Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n);
template <typename Dtype>
void col2octree_gpu(const Dtype* data_col, Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n);

void calc_neigh_cpu(int* neigh_split, const int* neigh,
    const int* children, const int node_num);
void calc_neigh_gpu(int* neigh_split, const int* neigh,
    const int* children, const int node_num);
void calc_neigh_cpu(int* neigh, const int depth, const int batch_size);
void calc_neigh_gpu(int* neigh, const int depth, const int batch_size);
// calculate neighborhood information with the hash table
void calc_neighbor(int* neigh, const unsigned* key, const int node_num,
    const int displacement = 0);

void generate_key_gpu(unsigned int* key_split, const unsigned int* key,
    const int* children, const int node_num);
void generate_key_cpu(unsigned int* key_split, const unsigned int* key,
    const int* children, const int node_num);
void generate_key_gpu(unsigned int* key, const int depth, const int batch_size);
void generate_key_cpu(unsigned int* key, const int depth, const int batch_size);

void xyz2key_cpu(unsigned int* key, const unsigned int* xyz, const int num, const int depth);
void xyz2key_gpu(unsigned int* key, const unsigned int* xyz, const int num, const int depth);

template <typename Dtype>
void generate_label_cpu(int* label_data, int& top_h, const Dtype* bottom_data,
    const int bottom_h, const int mask);
template <typename Dtype>
void generate_label_gpu(int* label_data, int& top_h, const Dtype* bottom_data,
    const int bottom_h, const int mask);

inline void compute_key(unsigned int& key, const unsigned int* pt, const int depth);
inline void compute_pt(unsigned int* pt, const unsigned int& key, const int depth);

void octree_dropout(vector<char>& octree_output, const string& octree_input,
    const int depth_dropout, const float threshold = 0.5f, const int channel = 3);
void aoctree_dropout(vector<char>& octree_output, const string& octree_input,
    const int depth_dropout, const int channel = 3);

template <typename Dtype>
void merge_octrees(Blob<Dtype>& octree_output,  const vector<vector<char> >& octrees);

template <typename Dtype>
void set_octree_parser(OctreeParser& octree_parser, const Blob<Dtype>& octree_in);

void search_key_cpu(int* idx, const int unsigned* key, const int n_key,
    const unsigned int* query, const int n_query);
void search_key_gpu(int* idx, const int unsigned* key, const int n_key,
    const unsigned int* query, const int n_query);

int content_flag(string str);

} // namespace octree
} // namespace caffe

#endif // CAFFE_UTIL_OCTREE_HPP_
