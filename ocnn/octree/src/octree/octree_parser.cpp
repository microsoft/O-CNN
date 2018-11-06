#include "octree_parser.h"

#include <cstring>
#include <fstream>

#include "marching_cube.h"

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
  for (int i = 0; i < 3; pt[i++] = 0u);

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < 3; j++) {
      // bit mask
      uint32 mask = 1u << (3 * i + 2 - j);
      // put the bit to position i
      pt[j] |= (key & mask) >> (2 * i + 2 - j);
    }
  }
}

int OctreeParser::clamp(int val, const int val_min, const int val_max) {
  if (val < val_min) val = val_min;
  if (val > val_max) val = val_max;
  return val;
}

bool OctreeParser::read_octree(const string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) return false;

  infile.seekg(0, infile.end);
  size_t len = infile.tellg();
  infile.seekg(0, infile.beg);

  buffer_.resize(len);
  infile.read(buffer_.data(), len);
  info_ = reinterpret_cast<OctreeInfo*>(buffer_.data());

  infile.close();
  return true;
}

bool OctreeParser::write_octree(const string& filename) const {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) return false;
  outfile.write(buffer_.data(), buffer_.size());
  outfile.close();
  return true;
}

std::string OctreeParser::get_binary_string() const {
    return std::string(buffer_.cbegin(), buffer_.cend());
}

const char* OctreeParser::ptr(const OctreeInfo::PropType ptype, const int depth) const {
  const char* p = nullptr;
  int dis = info_->ptr_dis(ptype, depth);
  if (-1 != dis) {
    p = buffer_.data() + dis;
  }
  return p;
}

const unsigned int* OctreeParser::key(const int depth) const {
  return reinterpret_cast<const uint32*>(ptr(OctreeInfo::kKey, depth));
}

const int* OctreeParser::child(const int depth) const {
  return reinterpret_cast<const int*>(ptr(OctreeInfo::kChild, depth));
}

const int* OctreeParser::neigh(const int depth) const {
  return reinterpret_cast<const int*>(ptr(OctreeInfo::kNeigh, depth));
}

const float* OctreeParser::feature(const int depth) const {
  return reinterpret_cast<const float*>(ptr(OctreeInfo::kFeature, depth));
}

const float* OctreeParser::label(const int depth) const {
  return reinterpret_cast<const float*>(ptr(OctreeInfo::kLabel, depth));
}

const float* OctreeParser::split(const int depth) const {
  return reinterpret_cast<const float*>(ptr(OctreeInfo::kSplit, depth));
}

void OctreeParser::set_octree(vector<char>& data) {
  buffer_.swap(data);
  info_ = reinterpret_cast<OctreeInfo*>(buffer_.data());
}

void OctreeParser::set_octree(const char* data, const int sz) {
  resize_octree(sz);
  memcpy(buffer_.data(), data, sz);
}

void OctreeParser::resize_octree(const int sz) {
  buffer_.resize(sz);
  info_ = reinterpret_cast<OctreeInfo*>(buffer_.data());
}

char* OctreeParser::mutable_ptr(const OctreeInfo::PropType ptype, const int depth) {
  return const_cast<char*>(ptr(ptype, depth));
}

unsigned int* OctreeParser::mutable_key(const int depth) {
  return reinterpret_cast<uint32*>(mutable_ptr(OctreeInfo::kKey, depth));
}

int* OctreeParser::mutable_child(const int depth) {
  return reinterpret_cast<int*>(mutable_ptr(OctreeInfo::kChild, depth));
}

int* OctreeParser::mutable_neigh(const int depth) {
  return reinterpret_cast<int*>(mutable_ptr(OctreeInfo::kNeigh, depth));
}

float* OctreeParser::mutable_feature(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr(OctreeInfo::kFeature, depth));
}

float* OctreeParser::mutable_label(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr(OctreeInfo::kLabel, depth));
}

float* OctreeParser::mutable_split(const int depth) {
  return reinterpret_cast<float*>(mutable_ptr(OctreeInfo::kSplit, depth));
}

void OctreeParser::octree2pts(Points& point_cloud, int depth_start, int depth_end) {
  bool has_dis = info_->has_displace();
  const int depth = info_->depth();
  const int depth_full = info_->full_layer();
  const int depth_adpt = info_->adaptive_layer();
  const float kDis = 0.8660254f; // = sqrt(3.0f) / 2.0f
  const float* bbmin = info_->bbmin();
  const float kMul = info_->bbox_max_width() / float(1 << info_->depth());
  const bool key2xyz = info_->key2xyz();

  // update depth_start and depth_end
  depth_start = clamp(depth_start, depth_full, depth);
  if (info_->is_adaptive() && depth_start < depth_adpt) depth_start = depth_adpt;
  int location = info_->locations(OctreeInfo::kFeature);
  if (location != -1) depth_start = depth;
  depth_end = clamp(depth_end, depth_start, depth);

  vector<float> pts, normals, labels;
  for (int d = depth_start; d <= depth_end; ++d) {
    const uint32* key_d = key(d);
    const float* feature_d = feature(d);
    const int* child_d = child(d);
    const float* label_d = label(d);
    const int num = info_->nnum(d);
    const float scale = (1 << (depth - d)) * kMul;

    for (int i = 0; i < num; ++i) {
      float n[3], len = 0.0f;
      for (int c = 0; c < 3; ++c) {
        n[c] = feature_d[c * num + i];
        len += n[c] * n[c];
      }

      //if (node_type(pc[i]) == kLeaf) continue;
      if (len == 0 || (node_type(child_d[i]) != kLeaf && d != depth)) continue;

      uint32 pt[3] = { 0, 0, 0 };
      if (key2xyz) {
        // todo: when depth > 8, then the channel of key is 2
        const unsigned char* ptr = reinterpret_cast<const unsigned char*>(key_d + i);
        for (int c = 0; c < 3; ++c) { pt[c] = ptr[c]; }
      } else {
        compute_pt(pt, key_d[i], d);
      }

      for (int c = 0; c < 3; ++c) {
        float t = pt[c] + 0.5f;
        if (has_dis) {
          float dis = feature_d[3 * num + i] * kDis;
          t += dis * n[c];
        }
        t = t * scale + bbmin[c]; // !!! note the scale
        normals.push_back(n[c]);
        pts.push_back(t);
      }
      if (label_d != nullptr) labels.push_back(label_d[i]);
    }
  }

  point_cloud.set_points(pts, normals, vector<float>(), labels);
}

void OctreeParser::octree2mesh(vector<float>& V, vector<int>& F, int depth_start,
    int depth_end) {
  bool has_dis = info_->has_displace();
  const int depth = info_->depth();
  const int depth_full = info_->full_layer();
  const int depth_adpt = info_->adaptive_layer();
  const float kDis = 0.8660254f; // = sqrt(3.0f) / 2.0f
  const float* bbmin = info_->bbmin();
  const float kMul = info_->bbox_max_width() / float(1 << info_->depth());
  const bool key2xyz = info_->key2xyz();

  // update depth_start and depth_end
  depth_start = clamp(depth_start, depth_full, depth);
  if (info_->is_adaptive() && depth_start < depth_adpt) depth_start = depth_adpt;
  int location = info_->locations(OctreeInfo::kFeature);
  if (location != -1) depth_start = depth;
  depth_end = clamp(depth_end, depth_start, depth);

  V.clear(); F.clear();
  for (int d = depth_start; d <= depth_end; ++d) {
    const uint32* key_d = key(d);
    const float* feature_d = feature(d);
    const int* child_d = child(d);
    const int num = info_->nnum(d);
    const float scale = (1 << (depth - d)) * kMul;

    vector<float> pts, normals, labels, pts_ref;
    for (int i = 0; i < num; ++i) {
      float n[3], len = 0.0f;
      for (int c = 0; c < 3; ++c) {
        n[c] = feature_d[c * num + i];
        len += n[c] * n[c];
      }

      //if (node_type(pc[i]) == kLeaf) continue;
      if (len == 0 || (node_type(child_d[i]) != kLeaf && d != depth)) continue;

      uint32 pt[3] = { 0, 0, 0 };
      if (key2xyz) {
        // todo: when depth > 8, then the channel of key is 2
        const unsigned char* ptr = reinterpret_cast<const unsigned char*>(key_d + i);
        for (int c = 0; c < 3; ++c) { pt[c] = ptr[c]; }
      } else {
        compute_pt(pt, key_d[i], d);
      }
      for (int c = 0; c < 3; ++c) {
        float t = pt[c] + 0.5f;
        if (has_dis) {
          float dis = feature_d[3 * num + i] * kDis;
          t += dis * n[c];
        }

        //t = t * scale + bbmin[c]; // !!! note the scale
        normals.push_back(n[c]);
        pts.push_back(t);
        pts_ref.push_back(pt[c]);
      }
    }

    vector<float> vtx;
    vector<int> face;
    marching_cube_octree(vtx, face, pts, pts_ref, normals);


    // concate
    int nv = V.size() / 3;
    for (auto f : face) {
      F.push_back(f + nv);
    }

    // rescale the vtx and concatenated to V
    nv = vtx.size() / 3;
    for (int i = 0; i < nv; ++i) {
      for (int c = 0; c < 3; ++c) {
        float vl = vtx[i * 3 + c] * scale + bbmin[c];
        V.push_back(vl);
      }
    }
  }
}
