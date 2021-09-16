import os
import torch
import ocnn
import unittest


class OctreeKeyTest(unittest.TestCase):
  def test_decode_encode_key(self):
    samples = ocnn.octree_samples(['octree_1', 'octree_1'])
    octree = ocnn.octree_batch(samples).cuda()
    xyz = ocnn.octree_property(octree, 'xyz', 5)
    pts = ocnn.octree_decode_key(xyz)
    xyz_encode = ocnn.octree_encode_key(pts)

    gt = torch.cuda.ShortTensor([
        [16, 16, 16, 0], [16, 16, 17, 0], [16, 17, 16, 0], [16, 17, 17, 0],
        [17, 16, 16, 0], [17, 16, 17, 0], [17, 17, 16, 0], [17, 17, 17, 0],
        [16, 16, 16, 1], [16, 16, 17, 1], [16, 17, 16, 1], [16, 17, 17, 1],
        [17, 16, 16, 1], [17, 16, 17, 1], [17, 17, 16, 1], [17, 17, 17, 1]])
    self.assertTrue((gt == pts).cpu().numpy().all())
    self.assertTrue((xyz_encode == xyz).cpu().numpy().all())

  def test_xyz_key(self):
    samples = ocnn.octree_samples(['octree_1', 'octree_1'])
    octree = ocnn.octree_batch(samples).cuda()
    xyz = ocnn.octree_property(octree, 'xyz', 5)
    key = ocnn.octree_xyz2key(xyz, 5)
    xyz_out = ocnn.octree_key2xyz(key, 5)
    self.assertTrue((xyz == xyz_out).cpu().numpy().all())

  def test_search_key(self):
    samples = ocnn.octree_samples(['octree_1', 'octree_1'])
    octree = ocnn.octree_batch(samples).cuda()

    key = torch.cuda.LongTensor([28673, 281474976739335, 10])
    idx_gt = torch.cuda.IntTensor([1, 15, -1])
    idx = ocnn.octree_search_key(key, octree, 5, key_is_xyz=False, nempty=False)
    self.assertTrue((idx == idx_gt).cpu().numpy().all())

    key = torch.cuda.LongTensor([28672, 28673, 281474976739328, 10])
    idx_gt = torch.cuda.IntTensor([0, -1, 1, -1])
    idx = ocnn.octree_search_key(key, octree, 5, key_is_xyz=False, nempty=True)
    self.assertTrue((idx == idx_gt).cpu().numpy().all())

  def test_xyz_key_64(self):
    # the length of key is over 32 bits
    xyz = torch.cuda.ShortTensor([[2049, 4095, 8011, 1], [511, 4095, 8011, 0]])
    xyz_encode = ocnn.octree_encode_key(xyz)
    key = ocnn.octree_xyz2key(xyz_encode, 13)
    xyz_out = ocnn.octree_key2xyz(key, 13)
    xyz_decode = ocnn.octree_decode_key(xyz_out)
    self.assertTrue((xyz == xyz_decode).cpu().numpy().all())


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
