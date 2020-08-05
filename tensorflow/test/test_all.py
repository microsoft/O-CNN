import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from test_octree2col import Octree2ColTest
from test_octree_conv import OctreeConvTest
from test_octree_deconv import OctreeDeconvTest
from test_octree_property import OctreePropertyTest
from test_octree_search import OctreeSearchTest
from test_octree_linear import OctreeLinearTest
from test_octree_nearest import OctreeNearestTest
from test_octree_gather import OctreeGatherTest
from test_octree_align import OctreeAlignTest
from test_octree2points import Octree2PointsTest
from test_points_property import PointsPropertyTest


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()