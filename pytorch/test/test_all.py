import os
import unittest
from test_octree_conv import OctreeConvTest
from test_octree_deconv import OctreeDeconvTest
from test_octree2col import Octree2ColTest
from test_octree_pool import OctreePoolTest
from test_octree_property import OctreePropertyTest
from test_octree_key import OctreeKeyTest
from test_points_property import PointsPropertyTest
from test_octree_trilinear import OctreeTrilinearTest
from test_octree_align import OctreeAlignTest

# Run 16 test in total
if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
