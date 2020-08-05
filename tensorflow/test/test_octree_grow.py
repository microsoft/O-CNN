import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from libs import *


tf.compat.v1.enable_eager_execution()


depth, channel = 5, 3
octree = octree_new(batch_size=1, channel=channel, has_displace=False)
octree = octree_grow(octree, target_depth=1, full_octree=True)
octree = octree_grow(octree, target_depth=2, full_octree=True)

octree_gt = tf.decode_raw(octree_samples('octree_2'), out_type=tf.int8)
for d in range(2, depth + 1):
  child = octree_child(octree_gt, depth=d)
  label = tf.cast(child > -1, tf.int32)
  octree = octree_update(octree, label, depth=d, mask=1)
  if d < depth:
    octree = octree_grow(octree, target_depth=d + 1, full_octree=False)

signal = octree_signal(octree_gt, depth, channel)
octree = octree_set_property(octree, signal, property_name="feature", depth=depth)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
#   tf.summary.FileWriter('./logs', sess.graph)
#   o1, o2 = sess.run([octree_gt, octree])

o1 = octree_gt.numpy()
o2 = octree.numpy()

print('Please check the output files: `octree_1.octree` and `octree_2.octree`')
with open('octree_1.octree', 'wb') as f:
  f.write(o1.tobytes())
with open('octree_2.octree', 'wb') as f:
  f.write(o2.tobytes())
