#include "caffe/util/octree_info.hpp"

#include <cstring>

namespace caffe {

const char OctreeInfo::kMagicStr[16] = "_OCTREE_1.0_";

void OctreeInfo::reset() {
  memset(this, 0, sizeof(OctreeInfo));
  strcpy(magic_str_, kMagicStr);
}

bool OctreeInfo::check_format(string& msg) const {
  msg.clear();
  if (strcmp(kMagicStr, magic_str_) != 0) {
    msg += "The version of octree format is not " + string(kMagicStr) + ".\n";
  }
  if (batch_size_ < 0) {
    msg += "The batch_size_ should be larger than 0.\n";
  }
  if (depth_ < 1 || depth_ > 8) {
    msg += "The depth_ should be in range [1, 8].\n";
  }
  if (full_layer_ < 0 || full_layer_ > depth_) {
    msg += "The full_layer_ should be in range [1, depth_].\n";
  }
  if (adp_layer_ < full_layer_ || adp_layer_ > depth_) {
    msg += "The adp_layer_ should be in range [full_layer_, depth_].\n";
  }
  const int channel_max[] = { 2, 1, 8, 1 << 30, 1, 1 };
  for (int i = 0; i < kPTypeNum; ++i) {
    string str = std::to_string(i);
    if (channels_[i] < 0 && channels_[i] > channel_max[i]) {
      msg += "The channel " + str + " should be in range [0, " +
          std::to_string(channel_max[i]) + "].\n";
    }
    if ((channels_[i] == 0) != ((content_flags_ & (1 << i)) == 0)) {
      msg += "The content_flags_ should be consistent with channels_[" + str + "].\n";
    }
    if (channels_[i] != 0 && locations_[i] != -1 && locations_[i] != depth_) {
      msg += "The locations_[" + str + "] should be -1 or " + std::to_string(depth_) + ".\n";
    }
  }

  // the OctreeInfo is valid when no error message is produced
  return msg.empty();
}

bool OctreeInfo::is_consistent(const OctreeInfo& info) const {
  // ignore threshold_dist_, threshold_norm_, nnum_, nnum_cum_,
  // nnum_nempty_, bbmin_, bbmax_, ptr_dis_
  return strcmp(magic_str_, info.magic_str_) == 0 &&
      memcmp(channels_, info.channels_, 16 * sizeof(int)) == 0 &&
      memcmp(locations_, info.locations_, 16 * sizeof(int)) == 0 &&
      batch_size_ == info.batch_size_ && depth_ == info.depth_ &&
      full_layer_ == info.full_layer_ && adp_layer_ == info.adp_layer_ &&
      is_adaptive_ == info.is_adaptive_ && key2xyz_ == info.key2xyz_ &&
      has_node_dis_ == info.has_node_dis_ &&
      content_flags_ == info.content_flags_;
}

int OctreeInfo::channel(PropType ptype) const {
  if (!has_property(ptype)) return 0;
  int i = property_index(ptype);
  return channels_[i];
}

int OctreeInfo::locations(PropType ptype) const {
  if (!has_property(ptype)) return 0;
  int i = property_index(ptype);
  return locations_[i];
}

int OctreeInfo::ptr_dis(PropType ptype, const int depth) const {
  if (!has_property(ptype)) return -1;
  int i = property_index(ptype);
  int dis = ptr_dis_[i];
  if (locations(ptype) == -1) {
    // !!! Note: sizeof(int)
    dis += nnum_cum_[depth] * channel(ptype) * sizeof(int);
  } else {
    // ignore the input parameter depth
  }
  return dis;
}

float OctreeInfo::bbox_max_width() const {
  float max_width = bbmax_[0] - bbmin_[0];
  for (int i = 1; i < 3; ++i) {
    float dis = bbmax_[i] - bbmin_[i];
    if (dis > max_width) max_width = dis;
  }
  // deal with degenarated case
  if (max_width == 0.0f) max_width = 1.0e-10f;
  return max_width;
}

void OctreeInfo::set_batch_size(int b) {
  batch_size_ = b < 1 ? 1 : b;
}

void OctreeInfo::set_depth(int d) {
  depth_ = full_layer_ < d ? d : full_layer_;
}

void OctreeInfo::set_full_layer(int fd) {
  full_layer_ = fd < 1 ? 1 : fd;
}

void OctreeInfo::set_nnum(int d, int num) {
  nnum_[d] = num;
}

void OctreeInfo::set_nnum(const int* num) {
  memcpy(nnum_, num, sizeof(int) * (depth_ + 1));
}

void OctreeInfo::set_nempty(int d, int num) {
  nnum_nempty_[d] = num;
}

void OctreeInfo::set_nempty(const int* num) {
  memcpy(nnum_nempty_, num, sizeof(int) * (depth_ + 1));
}

void OctreeInfo::set_nnum_cum(int capacity) {
  nnum_cum_[0] = 0;
  for (int d = 1; d < depth_ + 2; ++d) {
    nnum_cum_[d] = nnum_cum_[d - 1] + nnum_[d - 1];
  }
  nnum_cum_[depth_ + 2] = capacity > nnum_cum_[depth_ + 1] ?
      capacity : nnum_cum_[depth_ + 1];
}

void OctreeInfo::set_property(PropType ptype, int ch, int lc) {
  // this is just a convenient interface to make sure that
  // the set_channel and set_location be called together
  set_channel(ptype, ch);
  set_location(ptype, lc);
}

void OctreeInfo::set_channel(PropType ptype, int ch) {
  // note: the channel and content_flags_ are consisent.
  // If channels_[i] != 0, then the i^th bit of content_flags_ is 1.
  int i = property_index(ptype);
  if (ch > 0) {
    channels_[i] = ch;
    content_flags_ |= ptype;
  } else {
    channels_[i] = 0;
    // set the corresponding content_flags_ bit as 0
    content_flags_ &= ~ptype;
  }
}

void OctreeInfo::set_location(PropType ptype, int lc) {
  // lc: -1, the property exists at all node
  // lc:  d, the property exists at the d^th level of the octree
  // So lc must be in the set [-1, depth]
  int i = property_index(ptype);
  locations_[i] = lc;
}

void OctreeInfo::set_ptr_dis() {
  // todo: Add the following code because I changed the defination
  // of set_nnum_cum() on 09/25/2018. After the testing, safely
  // removing the following code
  if (total_nnum_capacity() < total_nnum()) {
    set_nnum_cum();   // update the capacity
  }

  // the accumulated pointer displacement
  ptr_dis_[0] = sizeof(OctreeInfo);
  for (int i = 1; i <= kPTypeNum; ++i) { // note the " <= " is used here
    PropType ptype = static_cast<PropType>(1 << (i - 1));
    int lc = locations(ptype);
    int num = lc == -1 ? total_nnum_capacity() : node_num(lc);
    // If the property do not exist, lc is equal to 0, then num = 8, both of them
    // are meaningless. Their values are wiped out by channels_[i - 1] (= 0).
    // So the value of ptr_dis_[i] is still correct.
    // !!! Note: sizeof(int)
    ptr_dis_[i] = ptr_dis_[i - 1] + sizeof(int) * num * channels_[i - 1];
  }
}

void OctreeInfo::set_bbox(const float* bbmin, const float* bbmax) {
  const int dim = 3;
  for (int i = 0; i < dim; ++i) {
    bbmin_[i] = bbmin[i];
    bbmax_[i] = bbmax[i];
  }
}

int OctreeInfo::property_index(PropType ptype) const {
  int k = 0, p = ptype;
  for (int i = 0; i < kPTypeNum; ++i) {
    if (0 != (p & (1 << i))) {
      k = i; break;
    }
  }
  return k;
}

} // namespace caffe
