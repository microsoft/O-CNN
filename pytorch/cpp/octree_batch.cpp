#include <octree/merge_octrees.h>

#include "ocnn.h"

Tensor octree_batch(vector<Tensor> tensors_in) {
  int batch_size = tensors_in.size();
  vector<const char*> octrees_in;
  for (int i = 0; i < batch_size; ++i) {
    const char* ptr = (const char*)tensors_in[i].data_ptr<uint8_t>();
    octrees_in.push_back(ptr);
  }

  // merge octrees
  vector<char> octree_out;
  merge_octrees(octree_out, octrees_in);

  // copy output
  int64_t size = octree_out.size();
  Tensor tensor_out = torch::zeros({size}, torch::dtype(torch::kUInt8));
  memcpy(tensor_out.data_ptr<uint8_t>(), octree_out.data(), size);
  return tensor_out;
}
