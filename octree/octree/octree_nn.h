#ifndef _OCTREE_OCTREE_NN_
#define _OCTREE_OCTREE_NN_

#include <vector>
#include <string>
#include <unordered_map>

#include "types.h"

using std::string;
using std::vector;
using std::unordered_map;


// A singleton class to hold global & common stuff for octree neighbor
class NeighHelper {
 public:
  static NeighHelper& Get() {
    static NeighHelper instance; // Guaranteed to be destroyed.
    return instance;             // Instantiated on first use.
  }

  static vector<int>& get_parent_array() { return Get().parent_; }
  static vector<int>& get_dis_array() { return Get().displacement_; }
  static vector<int>& get_bilinear_array() { return Get().bilinear_; }
  static vector<int>& get_ni(const vector<int>& kernel_size);

 private:
  NeighHelper() { init_neigh_index(); }
  void init_neigh_index();

  // avoid accidentally getting copies of your singleton appearing.
  NeighHelper(NeighHelper const&);
  void operator=(NeighHelper const&);

 protected:
  // used to get the neighbor information
  vector<vector<int> > ni_;
  unordered_map<string, int> ni_map_;

  // used to calculate the neighbor information
  vector<int> parent_;
  vector<int> displacement_;

  // used to bilinear interpolation
  vector<int> bilinear_;
};


// !!! TODO: support gpu stream for gpu functions !!!

int num_elements(const vector<int>& shape);
template <typename Dtype>
void resize_with_last_val(vector<Dtype>& vec, const int size);

template <typename Dtype>
void memset_cpu(const size_t, const Dtype alpha, Dtype *X);
template <typename Dtype>
void memset_gpu(const size_t, const Dtype alpha, Dtype *X);
template <typename Dtype>
void memcpy_cpu(const size_t N, const Dtype* X, Dtype* Y);
template <typename Dtype>
void memcpy_gpu(const size_t N, const Dtype* X, Dtype* Y);


template<typename Dtype>
void pad_forward_cpu(Dtype* Y, const int Hy, const int Cy,
    const Dtype* X, const int Hx, const int* label, const Dtype dval = 0);
template<typename Dtype>
void pad_forward_gpu(Dtype* Y, const int Hy, const int Cy,
    const Dtype* X, const int Hx, const int* label, const Dtype dval = 0);
template<typename Dtype>
void pad_backward_cpu(Dtype* X, const int Hx, const int Cx,
    const Dtype* Y, const int Hy, const int* label);
template<typename Dtype>
void pad_backward_gpu(Dtype* X, const int Hx, const int Cx,
    const Dtype* Y, const int Hy, const int* label);


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

template <typename Dtype>
void octree2colP_cpu(Dtype* data_col, const Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n);
template <typename Dtype>
void octree2colP_gpu(Dtype* data_col, const Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n);
template <typename Dtype>
void col2octreeP_cpu(const Dtype* data_col, Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n);
template <typename Dtype>
void col2octreeP_gpu(const Dtype* data_col, Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n);

template <typename Dtype>
void octree_max_pool_cpu(Dtype* top_data, int top_h, int* mask,
    const Dtype* btm_data, int btm_h, int channel);
template <typename Dtype>
void octree_max_pool_gpu(Dtype* top_data, int top_h, int* mask,
    const Dtype* btm_data, int btm_h, int channel);
template <typename Dtype>
void octree_max_unpool_cpu(const Dtype* top_diff, int top_h, const int* mask,
    Dtype* btm_diff, int btm_h, int channel);
template <typename Dtype>
void octree_max_unpool_gpu(const Dtype* top_diff, int top_h, const int* mask,
    Dtype* btm_diff, int btm_h, int channel);
template <typename Dtype>
void octree_mask_pool_cpu(Dtype* top_data, int top_h, const int* mask,
    const Dtype* btm_data, int btm_h, int channel);
template <typename Dtype>
void octree_mask_pool_gpu(Dtype* top_data, int top_h, const int* mask,
    const Dtype* btm_data, int btm_h, int channel);


void calc_neigh_cpu(int* neigh_child, const int* neigh, const int* child,
    const int node_num);
void calc_neigh_gpu(int* neigh_child, const int* neigh, const int* children,
    const int node_num, const int* parent, const int* dis);
void calc_neigh_cpu(int* neigh, const int depth, const int batch_size);
void calc_neigh_gpu(int* neigh, const int depth, const int batch_size);

// !!! The generated key is in the format of zipped (x,y,z,id)
template <typename Dtype>
void generate_key_gpu(Dtype* key_child, const Dtype* key, const int* child,
    const int node_num);
template <typename Dtype>
void generate_key_cpu(Dtype* key_child, const Dtype* key, const int* child,
    const int node_num);
template <typename Dtype>
void generate_key_gpu(Dtype* key, const int depth, const int batch_size);
template <typename Dtype>
void generate_key_cpu(Dtype* key, const int depth, const int batch_size);


template <typename Dtype>
void generate_label_cpu(int* label_data, int& top_h, const Dtype* bottom_data,
    const int bottom_h, const int mask);
template <typename Dtype>
void generate_label_gpu(int* label_data, int& top_h, const Dtype* bottom_data,
    const int bottom_h, const int mask);


void bilinear_neigh_cpu(int* bidx, const int* neigh, const int* child,
    const int node_num, const int* table);
void bilinear_neigh_gpu(int* bidx, const int* neigh, const int* child,
    const int node_num, const int* table);
void bilinear_xyz_cpu(uintk* xyz0, float* dis, const int d0, const uintk* xyz1,
    const int d1, const int num);
void bilinear_xyz_gpu(uintk* xyz0, float* dis, const int d0, const uintk* xyz1,
    const int d1, const int num);


template <typename Dtype>
void sequence_cpu(Dtype* ptr, const int num);
template <typename Dtype>
void sequence_gpu(Dtype* ptr, const int num);


// TODO: The performance can be improved via descent along the octree
template <typename Dtype>
void search_key_cpu(int* idx, const Dtype* key, const int n_key,
    const Dtype* query, const int n_query);
template <typename Dtype>
void search_key_gpu(int* idx, const Dtype* key, const int n_key,
    const Dtype* query, const int n_query);


template <typename Dtype>
void align_forward_gpu(Dtype* top_data, const int top_h, const int channel,
    const Dtype* btm_data, const int btm_h, const int* idx);
template <typename Dtype>
void align_backward_gpu(const Dtype* top_data, const int top_h, const int channel,
    Dtype* btm_data, const int btm_h, const int* idx);


// TODO: The implementation of align_forward_gpu, octree_pad_gpu, and 
// octree_gather_gpu is very similar, try to merge these codes:
// 1. Merge octree_gather_gpu and octree_pad_gpu: 
//    the height of `idx` is `top_h` in octree_gather_gpu
// 2. Change the implementation of align_forward_gpu to make the height
//    of `idx` is `top_h` instead of `btm_h`
template <typename Dtype>
void octree_gather_gpu(Dtype* top_data, const int top_h, const int channel,
  const Dtype* btm_data, const int btm_h, const int* idx);
template <typename Dtype>
void octree_gatherbk_gpu(const Dtype* top_data, const int top_h, const int channel,
  Dtype* btm_data, const int btm_h, const int* idx);


void octree_mask_gpu(float* out_data, const float* in_data, const int* label,
    int height, int mask, int num);


// !!! Caveat: for the following two functions, pt and depth
// must be consistent, i.e pt must be in the range [0, 2^depth]^3
template <typename Dtype>
void compute_key(Dtype& key, const Dtype* pt, const int depth);
template <typename Dtype>
void compute_pt(Dtype* pt, const Dtype& key, const int depth);

template <typename Dtype>
void xyz2key_cpu(Dtype* key, const Dtype* xyz, const int num, const int depth);
template <typename Dtype>
void xyz2key_gpu(Dtype* key, const Dtype* xyz, const int num, const int depth);
template <typename Dtype>
void key2xyz_cpu(Dtype* xyz, const Dtype* key, const int num, const int depth);
template <typename Dtype>
void key2xyz_gpu(Dtype* xyz, const Dtype* key, const int num, const int depth);

template <typename Dtype>
void key2idx_cpu(int* idx, const Dtype* key, const int num);
template <typename Dtype>
void key2idx_gpu(int* idx, const Dtype* key, const int num);

template <typename Dtype>
void xyz2coord_cpu(float* pt, const Dtype* xyz, const int num, const int channel);
template <typename Dtype>
void xyz2coord_gpu(float* pt, const Dtype* xyz, const int num, const int channel);
template <typename Dtype>
void coord2xyz_cpu(Dtype* xyz, const float* pt, const int num, const int channel);
template <typename Dtype>
void coord2xyz_gpu(Dtype* xyz, const float* pt, const int num, const int channel);

template<typename Dtype1, typename Dtype2>
void key2xyz(Dtype1* xyz, const Dtype2 key, const int depth);

#endif // _OCTREE_OCTREE_NN_
