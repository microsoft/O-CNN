import os
import unittest
from test_octree_conv import OctreeConvTest
from test_octree2col import Octree2ColTest
from test_octree_pool import OctreePoolTest
from test_octree_property import OctreePropertyTest

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
