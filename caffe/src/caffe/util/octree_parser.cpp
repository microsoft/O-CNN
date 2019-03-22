#include "caffe/util/octree_parser.hpp"

#include "caffe/common.hpp"

namespace caffe {

void OctreeParser::set_cpu(const void* ptr) {
  const_ptr_ = true;
  h_metadata_ = reinterpret_cast<char*>(const_cast<void*>(ptr));
  info_ = reinterpret_cast<OctreeInfo*>(h_metadata_);
}

void OctreeParser::set_cpu(void* ptr, OctreeInfo* octinfo) {
  const_ptr_ = false;
  h_metadata_ = reinterpret_cast<char*>(ptr);
  info_ = reinterpret_cast<OctreeInfo*>(ptr);
  if (octinfo != nullptr) { // update the OctreeInfo with octinfo
    memcpy(info_, octinfo, sizeof(OctreeInfo));
  }
}

#ifndef CPU_ONLY
void OctreeParser::set_gpu(const void* ptr, const void* oct_info) {
  const_ptr_ = true;
  d_metadata_ = reinterpret_cast<char*>(const_cast<void*>(ptr));

  // oct_info is a host pointer
  if (oct_info == nullptr) {
    info_ = &info_buffer_;
    CUDA_CHECK(cudaMemcpy(info_, ptr, sizeof(OctreeInfo), cudaMemcpyDeviceToHost));
  } else {
    info_ = reinterpret_cast<OctreeInfo*>(const_cast<void*>(oct_info));
  }
}

void OctreeParser::set_gpu(void* ptr, OctreeInfo* octinfo) {
  const_ptr_ = false;
  d_metadata_ = reinterpret_cast<char*>(ptr);
  info_ = &info_buffer_;
  if (octinfo != nullptr) { // update the OctreeInfo with octinfo
    memcpy(info_, octinfo, sizeof(OctreeInfo));
    CUDA_CHECK(cudaMemcpy(d_metadata_, info_, sizeof(OctreeInfo), cudaMemcpyHostToDevice));
  } else {
    CUDA_CHECK(cudaMemcpy(info_, d_metadata_, sizeof(OctreeInfo), cudaMemcpyDeviceToHost));
  }
}

#else

void OctreeParser::set_gpu(const void* ptr, const void* oct_info) {
  NO_GPU;
}

void OctreeParser::set_gpu(void* ptr, OctreeInfo* octinfo) {
  NO_GPU;
}

#endif


OctreeParser::NodeType OctreeParser::node_type(const int t) const {
  NodeType ntype = kInternelNode;
  if (t == -1) ntype = kLeaf;
  if (t == -2) ntype = kNonEmptyLeaf;
  return ntype;
}

void OctreeParser::compute_key(uint32& key, const uint32* pt, const int depth) {
  key = 0;
  for (int i = 0; i < depth; i++) {
    uint32 mask = 1u << i;
    for (int j = 0; j < 3; j++) {
      key |= (pt[j] & mask) << (2 * i + 2 - j);
    }
  }
}

void OctreeParser::compute_pt(uint32* pt, const uint32& key, const int depth) {
  // init
  for (int i = 0; i < 3; pt[i++] = 0u);

  // convert
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < 3; j++) {
      // bit mask
      uint32 mask = 1u << (3 * i + 2 - j);
      // put the bit to position i
      pt[j] |= (key & mask) >> (2 * i + 2 - j);
    }
  }
}

const char* OctreeParser::ptr_cpu(const PropType ptype, const int depth) const {
  CHECK(h_metadata_ != nullptr);
  const char* p = nullptr;
  int dis = info_->ptr_dis(ptype, depth);
  if (-1 != dis) {
    p = h_metadata_ + dis;
  }
  return p;
}

const unsigned int* OctreeParser::key_cpu(const int depth) const {
  return reinterpret_cast<const unsigned int*>(ptr_cpu(OctreeInfo::kKey, depth));
}

const int* OctreeParser::children_cpu(const int depth) const {
  return reinterpret_cast<const int*>(ptr_cpu(OctreeInfo::kChild, depth));
}

const int* OctreeParser::neighbor_cpu(const int depth) const {
  return reinterpret_cast<const int*>(ptr_cpu(OctreeInfo::kNeigh, depth));
}

const float* OctreeParser::feature_cpu(const int depth) const {
  return reinterpret_cast<const float*>(ptr_cpu(OctreeInfo::kFeature, depth));
}

const float* OctreeParser::label_cpu(const int depth) const {
  return reinterpret_cast<const float*>(ptr_cpu(OctreeInfo::kLabel, depth));
}

const float* OctreeParser::split_cpu(const int depth) const {
  return reinterpret_cast<const float*>(ptr_cpu(OctreeInfo::kSplit, depth));
}

char* OctreeParser::mutable_ptr_cpu(const OctreeInfo::PropType ptype, const int depth) {
  CHECK(const_ptr_ == false);
  return const_cast<char*>(ptr_cpu(ptype, depth));
}

unsigned int* OctreeParser::mutable_key_cpu(const int depth) {
  return reinterpret_cast<unsigned int*>(mutable_ptr_cpu(OctreeInfo::kKey, depth));
}

int* OctreeParser::mutable_children_cpu(const int depth) {
  return reinterpret_cast<int*>(mutable_ptr_cpu(OctreeInfo::kChild, depth));
}

int* OctreeParser::mutable_neighbor_cpu(const int depth) {
  return reinterpret_cast<int*>(mutable_ptr_cpu(OctreeInfo::kNeigh, depth));
}

float* OctreeParser::mutable_feature_cpu(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr_cpu(OctreeInfo::kFeature, depth));
}

float* OctreeParser::mutable_label_cpu(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr_cpu(OctreeInfo::kLabel, depth));
}

float* OctreeParser::mutable_split_cpu(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr_cpu(OctreeInfo::kSplit, depth));
}

const char* OctreeParser::ptr_gpu(const PropType ptype, const int depth) const {
  CHECK(d_metadata_ != nullptr);
  const char* p = nullptr;
  int dis = info_->ptr_dis(ptype, depth);
  if (-1 != dis) {
    p = d_metadata_ + dis;
  }
  return p;
}

const unsigned int* OctreeParser::key_gpu(const int depth) const {
  return reinterpret_cast<const unsigned int*>(ptr_gpu(OctreeInfo::kKey, depth));
}

const int* OctreeParser::children_gpu(const int depth) const {
  return reinterpret_cast<const int*>(ptr_gpu(OctreeInfo::kChild, depth));
}

const int* OctreeParser::neighbor_gpu(const int depth) const {
  return reinterpret_cast<const int*>(ptr_gpu(OctreeInfo::kNeigh, depth));
}

const float* OctreeParser::feature_gpu(const int depth) const {
  return reinterpret_cast<const float*>(ptr_gpu(OctreeInfo::kFeature, depth));
}

const float* OctreeParser::label_gpu(const int depth) const {
  return reinterpret_cast<const float*>(ptr_gpu(OctreeInfo::kLabel, depth));
}

const float* OctreeParser::split_gpu(const int depth) const {
  return reinterpret_cast<const float*>(ptr_gpu(OctreeInfo::kSplit, depth));
}

char* OctreeParser::mutable_ptr_gpu(const OctreeInfo::PropType ptype, const int depth) {
  CHECK(const_ptr_ == false);
  return const_cast<char*>(ptr_gpu(ptype, depth));
}

unsigned int* OctreeParser::mutable_key_gpu(const int depth) {
  return reinterpret_cast<unsigned int*>(mutable_ptr_gpu(OctreeInfo::kKey, depth));
}

int* OctreeParser::mutable_children_gpu(const int depth) {
  return reinterpret_cast<int*>(mutable_ptr_gpu(OctreeInfo::kChild, depth));
}

int* OctreeParser::mutable_neighbor_gpu(const int depth) {
  return reinterpret_cast<int*>(mutable_ptr_gpu(OctreeInfo::kNeigh, depth));
}

float* OctreeParser::mutable_feature_gpu(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr_gpu(OctreeInfo::kFeature, depth));
}

float* OctreeParser::mutable_label_gpu(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr_gpu(OctreeInfo::kLabel, depth));
}

float* OctreeParser::mutable_split_gpu(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr_gpu(OctreeInfo::kSplit, depth));
}

} // namespace caffe
