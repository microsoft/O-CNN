from libs import *
import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append('..')


class PointsPropertyTest(tf.test.TestCase):

  def test_points_property(self):
    points = np.arange(15, dtype=np.float32).reshape([-1, 3])
    index = np.array([[0.0], [0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    points4 = np.concatenate([points, index], axis=1)
    normals = np.arange(0, 1.5, 0.1, dtype=np.float32).reshape([-1, 3])
    labels = np.arange(5, dtype=np.float32)
    feats = np.arange(10, dtype=np.float32).reshape([-1, 2])

    # creat points
    pts1 = points_new(points[:3, :], normals[:3, :], feats[:3, :], labels[:3])
    pts2 = points_new(points[3:, :], normals[3:, :], feats[3:, :], labels[3:])

    # # save points
    # with open('points1.points', 'wb') as fid:
    #   fid.write(pts1.numpy()[0])
    # with open('points2.points', 'wb') as fid:
    #   fid.write(pts2.numpy()[0])

    # get property
    tf_labels = points_property([pts1, pts2], property_name='label', channel=1)
    tf_labels = tf.reshape(tf_labels, [-1])
    tf_normals = points_property([pts1, pts2], property_name='normal', channel=3)
    tf_pts3 = points_property([pts1, pts2], property_name='xyz', channel=3)
    tf_pts4 = points_property([pts1, pts2], property_name='xyz', channel=4)
    tf_feats = points_property([pts1, pts2], property_name='feature', channel=2)

    # set property
    points_set = -np.arange(9, dtype=np.float32).reshape([-1, 3])
    pts3 = points_set_property(pts1, points_set, property_name='xyz')
    tf_points_set = points_property(pts3, property_name='xyz', channel=3)
    feats_set = -np.arange(6, dtype=np.float32).reshape([-1, 2])
    tf_feats_set = points_property(pts3, property_name='feature', channel=2)

    with self.cached_session() as sess:
      self.assertAllEqual(tf_labels, labels)
      self.assertAllEqual(tf_normals, normals)
      self.assertAllEqual(tf_pts3, points)
      self.assertAllEqual(tf_pts4, points4)
      self.assertAllEqual(tf_feats, feats)
      self.assertAllEqual(tf_points_set, points_set)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()
