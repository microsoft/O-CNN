#include "octree_parser.h"
#include "device_alternate.h"

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

const char* OctreeParser::ptr_gpu(const PropType ptype, const int depth) const {
  CHECK(d_metadata_ != nullptr);
  const char* p = nullptr;
  int dis = info_->ptr_dis(ptype, depth);
  if (-1 != dis) {
    p = d_metadata_ + dis;
  }
  return p;
}

const uintk* OctreeParser::key_gpu(const int depth) const {
  return reinterpret_cast<const uintk*>(ptr_gpu(OctreeInfo::kKey, depth));
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

uintk* OctreeParser::mutable_key_gpu(const int depth) {
  return reinterpret_cast<uintk*>(mutable_ptr_gpu(OctreeInfo::kKey, depth));
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