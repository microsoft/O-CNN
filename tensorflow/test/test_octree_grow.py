import os
import sys
import numpy as np
# from tqdm import tqdm
import tensorflow as tf
sys.path.append("..")
from config import *
from ocnn import *


# # =====================
# tf.enable_eager_execution()

# octree = octree_new(batch_size=1, channel=4, has_displace=True)

# octree = octree_grow(octree, target_depth=1, full_octree=True)
# print('octree_1.octree')
# with open('octree_1.octree', 'wb') as f:
#   f.write(octree.numpy().tobytes())

# octree = octree_grow(octree, target_depth=2, full_octree=True)
# print('octree_2.octree')
# with open('octree_2.octree', 'wb') as f:
#   f.write(octree.numpy().tobytes())

# # load test octree
# depth = 6
# with open('octree_test.octree', 'rb') as f:
#   bytes_in = f.read()
# octree_gt = tf.decode_raw(bytes=bytes_in, out_type=tf.int8)
# label_gt = [None]*10
# for d in range(2, depth + 1):
#   label = octree_property(octree_gt, property_name="split", dtype=tf.float32, 
#                           depth=d, channel=1)
#   label_gt[d] = tf.reshape(tf.cast(label, dtype=tf.int32), [-1])

# for d in range(2, depth + 1):
#   octree = octree_update(octree, label_gt[d], depth=d, mask=1)
#   if d < depth:
#     octree = octree_grow(octree, target_depth=d+1, full_octree=False)

# print('octree_6.octree')
# with open('octree_6.octree', 'wb') as f:
#   f.write(octree.numpy().tobytes())


# ====================
# tf.enable_eager_execution()


depth  = 6
octree = octree_new(batch_size=1, channel=4, has_displace=True)
octree = octree_grow(octree, target_depth=1, full_octree=True)
octree = octree_grow(octree, target_depth=2, full_octree=True)

# load test octree
test_set = '/home/penwan/workspace/ps/Dataset/completion_test_octree_6_lmdb_new.tfrecord'
octree_gt, _ = dataset(test_set, 1)
label_gt = [None]*10
for d in range(2, depth + 1):
  label = octree_property(octree_gt, property_name="split", dtype=tf.float32, 
                          depth=d, channel=1)
  label_gt[d] = tf.reshape(tf.cast(label, dtype=tf.int32), [-1])

for d in range(2, depth + 1):
  octree = octree_update(octree, label_gt[d], depth=d, mask=1)
  if d < depth:
    octree = octree_grow(octree, target_depth=d+1, full_octree=False)

# d = 2
# octree = octree_update(octree, label_gt[d], depth=d, mask=1)
# octree = octree_grow(octree, target_depth=d+1, full_octree=False)

# d = 3
# octree = octree_update(octree, label_gt[d], depth=d, mask=1)
# octree = octree_grow(octree, target_depth=d+1, full_octree=False)

# d = 4
# octree = octree_update(octree, label_gt[d], depth=d, mask=1)
# octree = octree_grow(octree, target_depth=d+1, full_octree=False)

# d = 5
# octree = octree_update(octree, label_gt[d], depth=d, mask=1)
# octree = octree_grow(octree, target_depth=d+1, full_octree=False)

# d = 6
# octree = octree_update(octree, label_gt[d], depth=d, mask=1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  tf.summary.FileWriter('./logs', sess.graph)
  o1, o2 = sess.run([octree_gt, octree])

# o1 = octree_gt.numpy()
# o2 = octree.numpy()

print('output')
with open('octree_1.octree', 'wb') as f:
  f.write(o1.tobytes())
with open('octree_2.octree', 'wb') as f:
  f.write(o2.tobytes())
