#include <gtest/gtest.h>
#include <cmath>
#include <octree_nn.h>
#include <octree_samples.h>
#include <merge_octrees.h>
#include <types.h>


TEST(VecResizeTest, TestVecResize) {
  vector<int> vec0{ 3 }, gt0{ 3, 3, 3 };
  resize_with_last_val(vec0, 3);
  ASSERT_EQ(vec0, gt0);

  vector<int> vec1{ 3, 1, 1, 3 }, gt1{3, 1, 1};
  resize_with_last_val(vec1, 3);
  ASSERT_EQ(vec1, gt1);

  vector<int> vec2, gt2;
  resize_with_last_val(vec2, 3);
  ASSERT_EQ(vec2, gt2);
}

TEST(BiliearNeigh, TestBiliearNeigh) {
  const char* octree1 = (const char*) octree::get_one_octree("octree_1");
  vector<char> buffer;
  merge_octrees(buffer, vector<const char*> {octree1});

  OctreeParser parser;
  parser.set_cpu(buffer.data());

  int depth = 4;
  vector<int> bidx(parser.info().node_num(depth + 1) * 8);
  const int* child0 = parser.children_cpu(depth);
  const int nnum0 = parser.info().node_num(depth);
  const int nnum1 = parser.info().node_num(depth + 1);
  bilinear_neigh_cpu(bidx.data(), parser.neighbor_cpu(depth),
      child0, nnum0, NeighHelper::get_bilinear_array().data());

  // check
  typedef typename KeyTrait<uintk>::uints uints;
  const int weights[8] = { 27, 9, 9, 9, 3, 3, 3, 1 };
  const uintk* key0 = parser.key_cpu(depth);
  const uintk* key1 = parser.key_cpu(depth + 1);
  ASSERT_TRUE(parser.info().is_key2xyz());
  auto key_to_xyz = [](float * xyz, const uintk * key) {
    const uints* ptr = (const uints*)key;
    for (int i = 0; i < 3; ++i) { xyz[i] = (float)ptr[i] + 0.5f; }
  };
  for (int i = 0; i < nnum1; ++i) {
    float xyz0[3], xyz1[3];
    key_to_xyz(xyz1, key1 + i);
    for (int c = 0; c < 3; ++c) {
      xyz1[c] /= 2.0f;
    }

    for (int k = 0; k < 8; ++k) {
      int j = bidx[i * 8 + k];
      if (j < 0) continue;
      key_to_xyz(xyz0, key0 + j);

      int weight = 1;
      for (int c = 0; c < 3; ++c) {
        float dis = fabsf(xyz0[c] - xyz1[c]);
        ASSERT_LT(dis, 1.0f);
        weight *= (int)((1 - dis) * 4);
      }
      ASSERT_EQ(weight, weights[k]);
    }
  }

  // check
  vector<uintk> xyz10(nnum1 * 8), key10(nnum1 * 8), key_octree(nnum0);
  vector<float> fracs(nnum1 * 3);
  bilinear_xyz_cpu(xyz10.data(), fracs.data(), depth,
      parser.key_cpu(depth + 1), depth + 1, nnum1);
  xyz2key_cpu(key10.data(), xyz10.data(), xyz10.size(), depth);
  xyz2key_cpu(key_octree.data(), key0, nnum0, depth);
  vector<int> sidx(nnum1 * 8);
  search_key_cpu(sidx.data(), key_octree.data(), key_octree.size(),
      key10.data(), key10.size());
  for (int i = 0; i < sidx.size(); ++i) {
    ASSERT_EQ(sidx[i], bidx[i]);
  }
}

TEST(Coord2xyzTest, TestCoord2xyz) {
  typedef typename KeyTrait<uintk>::uints uints;

  float coord[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 };
  uints xyz[] = { 1, 3, 5, 7, 2, 4, 6, 8 };
  uintk rst[2] = { 0 };
  coord2xyz_cpu(rst, coord, 2, 4);
  uints* ptr = (uints*)rst;
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(ptr[i], xyz[i]);
  }
}