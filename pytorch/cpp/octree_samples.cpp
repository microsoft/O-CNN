#include <octree/octree_samples.h>

#include "ocnn.h"

vector<Tensor> octree_samples(vector<string> names) {
  int num = names.size();
  vector<Tensor> tensors;
  for (int i = 0; i < num; ++i) {
    size_t size = 0;
    const unsigned char* str = octree::get_one_octree(names[i].c_str(), &size);

    CHECK_GT(size, 0) << "The specified octree does not exit : " << names[i];
    Tensor tensor = torch::zeros({(int64_t)size}, torch::dtype(torch::kUInt8));
    memcpy(tensor.data_ptr<uint8_t>(), str, size);
    tensors.push_back(tensor);
  }
  return tensors;
}
