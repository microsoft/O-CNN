import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append('..')
from libs import *
import pyoctree

tf.enable_eager_execution()

filename1 = '/home/ps/workspace/ocnn-tf/script/logs/seg/0430_run_dataset/00.points'
filename2 = '/home/ps/workspace/ocnn-tf/script/logs/seg/0430_run_dataset/01.points'

with open(filename1, 'rb') as fid: 
  points_raw1 = fid.read()
with open(filename2, 'rb') as fid: 
  points_raw2 = fid.read()
points_raw = [points_raw1, points_raw2]

tf_xyz = points_property(points_raw, property_name='xyz', channel=4)
tf_label = points_property(points_raw, property_name='label', channel=1)

points1 = pyoctree.Points()
succ = points1.read_points(filename1)
points2 = pyoctree.Points()
succ = points2.read_points(filename2)

np_xyz1 = np.array(points1.points()).reshape((-1, 3))
np_xyz2 = np.array(points2.points()).reshape((-1, 3))
np_label1 = np.array(points1.labels()).reshape((-1, 1))
np_label2 = np.array(points2.labels()).reshape((-1, 1))
np_id1 = np.zeros(np_label1.shape)
np_id2 = np.ones(np_label2.shape)

np_xyz = np.concatenate(
    (np.concatenate((np_xyz1, np_id1), axis=1),
     np.concatenate((np_xyz2, np_id2), axis=1)), axis=0).astype(np.float32)
np_label = np.concatenate((np_label1, np_label2), axis=0).astype(np.float32)

eq1 = np.array_equal(np_xyz, tf_xyz.numpy())
eq2 = np.array_equal(np_label, tf_label.numpy())

print('Pass xyz test: ', eq1)
print('Pass label test: ', eq2)
