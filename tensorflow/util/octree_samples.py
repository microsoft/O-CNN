import sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from libs import octree_samples

tf.enable_eager_execution()

# dump the octree samples
for i in range(1, 7):
  name = 'octree_%d' % i
  octree = octree_samples(name)
  with open(name + '.octree', 'wb') as fid:
    fid.write(octree.numpy())
