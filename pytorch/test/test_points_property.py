import os
import torch
import ocnn
import unittest
import numpy as np


class PointsPropertyTest(unittest.TestCase):

  def test_points_property(self):
    points = torch.arange(15, dtype=torch.float32).reshape([-1, 3])
    index = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)
    points4 = torch.cat([points, index], axis=1)
    normals = torch.arange(0, 1.5, 0.1, dtype=torch.float32).reshape([-1, 3])
    labels = torch.arange(5, dtype=torch.float32)
    feats = torch.arange(10, dtype=torch.float32).reshape([-1, 2])

    # creat points
    pts1 = ocnn.points_new(points[:3, :], normals[:3, :], feats[:3, :], labels[:3])
    pts2 = ocnn.points_new(points[3:, :], normals[3:, :], feats[3:, :], labels[3:])

    # get batch property
    t_batch_labels = ocnn.points_batch_property([pts1, pts2], 'label').view(-1)
    t_batch_normals = ocnn.points_batch_property([pts1, pts2], 'normal')
    t_batch_pts3 = ocnn.points_batch_property([pts1, pts2], 'xyz')
    t_batch_pts4 = ocnn.points_batch_property([pts1, pts2], 'xyzi')
    t_batch_feats = ocnn.points_batch_property([pts1, pts2], 'feature')

    self.assertTrue((t_batch_labels == labels).numpy().all())
    self.assertTrue((t_batch_normals == normals).numpy().all())
    self.assertTrue((t_batch_pts3 == points).numpy().all())
    self.assertTrue((t_batch_pts4 == points4).numpy().all())
    self.assertTrue((t_batch_feats == feats).numpy().all())

    # get property
    t_pts = ocnn.points_property(pts1, 'xyz')
    t_normals = ocnn.points_property(pts1, 'normal')
    t_feats = ocnn.points_property(pts1, 'feature')
    t_labels = ocnn.points_property(pts1, 'label').view(-1)

    self.assertTrue((t_pts == points[:3, :]).numpy().all())
    self.assertTrue((t_normals == normals[:3, :]).numpy().all())
    self.assertTrue((t_feats == feats[:3, :]).numpy().all())
    self.assertTrue((t_labels == labels[:3]).numpy().all())

    # set property
    points_set = -torch.arange(9, dtype=torch.float32).reshape([-1, 3])
    pts3 = ocnn.points_set_property(pts1, points_set, 'xyz')
    t_points_set = ocnn.points_property(pts3, 'xyz')
    self.assertTrue((t_points_set == points_set).numpy().all())

    feats_set = -torch.arange(6, dtype=torch.float32).reshape([-1, 2])
    pts4 = ocnn.points_set_property(pts3, feats_set, 'feature')
    t_feats_set = ocnn.points_property(pts4, 'feature')
    self.assertTrue((t_feats_set == feats_set).numpy().all())


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
