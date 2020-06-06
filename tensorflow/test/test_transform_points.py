import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append('..')
from libs import *

tf.enable_eager_execution()


filename = 'scene0000_00_000.points'
with open(filename, 'rb') as fid: 
  points = fid.read()


radius, center = bounding_sphere(points)
print('radius: ', radius, ', center: ', center)
