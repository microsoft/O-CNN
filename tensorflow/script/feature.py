import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from dataset import DatasetFactory
from network_hrnet import HRNet
from config import parse_args
from ocnn import octree_property

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


FLAGS = parse_args()

## graph
octree, label = DatasetFactory(FLAGS.DATA.test)()
hrnet = HRNet(FLAGS.MODEL)
tensors = hrnet.network(octree, training=False)
with tf.variable_scope('signal'):
  child = octree_property(octree, property_name='child', dtype=tf.int32,
                          depth=FLAGS.DATA.test.depth, channel=1)
  child = tf.reshape(child, [-1])


## output
FLAGSS = FLAGS.SOLVER
test_iter = FLAGSS.test_iter
output_prefix = FLAGSS.logdir + '/' + os.path.basename(FLAGS.DATA.test.location)

# classification features
fc1 = tensors['fc1']
channel1 = int(fc1.shape[1])
fc1 = tf.reshape(fc1, [channel1])
features1 = np.zeros((test_iter, channel1), dtype=np.float32)

fc2 = tensors['fc2']
fc2 = tf.nn.l2_normalize(fc2, axis=1)
channel2 = int(fc2.shape[1])
fc2 = tf.reshape(fc2, [channel2])
features2 = np.zeros((test_iter, channel2), dtype=np.float32)

# segmentation features
seg = tensors['logit_seg'] # N x C
seg = tf.nn.l2_normalize(seg, axis=1)

# label
labels = np.zeros((test_iter))


## run
def cls_features(sess):
  print('Classification features...')
  for i in tqdm(range(0, test_iter), ncols=80):
    f1, f2, l = sess.run([fc1, fc2, label])
    features1[i, :] = f1
    features2[i, :] = f2
    labels[i] = l
  
  np.save(output_prefix + '_fc1', features1)
  np.save(output_prefix + '_fc2', features2)
  np.save(output_prefix + '_label', labels)

def seg_features(sess):
  print('Segmentation features...')
  for i in tqdm(range(0, test_iter), ncols=80):
    s, o, c, f = sess.run([seg, octree, child, fc2])

    o.tofile(output_prefix + '_%03d.octree' % i)
    np.save(output_prefix + '_%03d.seg.npy' % i, s)
    np.save(output_prefix + '_%03d.child.npy' % i, c)
    np.save(output_prefix + '_%03d.fc2.npy' % i, f)

assert(FLAGSS.ckpt)
tf_saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  tf.summary.FileWriter(FLAGSS.logdir, sess.graph)
  print('Restore from checkpoint: ', FLAGSS.ckpt)
  tf_saver.restore(sess, FLAGSS.ckpt)

  if FLAGSS.run == 'cls':
    cls_features(sess)
  elif FLAGSS.run == 'seg':
    seg_features(sess)
  else:
    print('Error, unsupported SOLVER.run: ' + FLAGSS.run)
