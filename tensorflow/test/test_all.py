import os
import tensorflow as tf

from test_octree2col import Octree2ColTest
from test_octree_conv import OctreeConvTest
from test_octree_deconv import OctreeDeconvTest
from test_octree_property import OctreePropertyTest
from test_octree_search import OctreeSearchTest


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()