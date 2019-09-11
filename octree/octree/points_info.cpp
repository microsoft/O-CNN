#include "points_info.h"
#include <cstring>

const char PointsInfo::kMagicStr[16] = "_POINTS_1.0_";

void PointsInfo::reset() {
  memset(this, 0, sizeof(PointsInfo));
  strcpy(magic_str_, kMagicStr);
}

bool PointsInfo::check_format(string& msg) const {
  msg.clear();
  if (strcmp(kMagicStr, magic_str_) != 0) {
    msg += "The version of points format is not " + string(kMagicStr) + ".\n";
  }
  if (pt_num_ <= 0) {
    msg += "The pt_num_ should be larger than 0.\n";
  }
  // todo: add more checks

  // the PtsInfo is valid when no error message is produced
  return msg.empty();
}

int PointsInfo::channel(PropType ptype) const {
  int i = property_index(ptype);
  if (!has_property(ptype)) return 0;
  return channels_[i];
}

void PointsInfo::set_channel(PropType ptype, const int ch) {
  // note: the channel and content_flags_ are consisent.
  // If channels_[i] != 0, then the i^th bit of content_flags_ is 1.
  int i = property_index(ptype);
  channels_[i] = ch;
  content_flags_ |= ptype;
}

int PointsInfo::ptr_dis(PropType ptype) const {
  int i = property_index(ptype);
  if (!has_property(ptype)) return -1;
  return ptr_dis_[i];
}

void PointsInfo::set_ptr_dis() {
  // the accumulated pointer displacement
  ptr_dis_[0] = sizeof(PointsInfo);
  for (int i = 1; i <= kPTypeNum; ++i) { // note the " <= " is used here
    ptr_dis_[i] = ptr_dis_[i - 1] + sizeof(float) * pt_num_ * channels_[i - 1];
  }
}

int PointsInfo::property_index(PropType ptype) const {
  int k = 0, p = ptype;
  for (int i = 0; i < kPTypeNum; ++i) {
    if (0 != (p & (1 << i))) {
      k = i; break;
    }
  }
  return k;
}
