"""Classes for creating and augmenting Octrees"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from ocnn.dataset.data_processor import DataProcessor
from ocnn.octree._octree import Octree
from ocnn.octree._octree import OctreeInfo
from ocnn.octree._octree import Points

#TODO Reduce calls to get_points_bounds for augmentors.

class OctreeSettings:
    """ Octree settings variables"""
    def __init__(self,
                 depth=6,
                 full_depth=2,
                 node_displacement=True,
                 node_feature=False,
                 split_label=False,
                 adaptive=False,
                 adaptive_depth=4,
                 threshold_distance=0.866,
                 threshold_normal=0.2,
                 key2xyz=False):
        """ Initializes OctreeSettings
        Args:
          depth: Maximum depth of the octree.
          full_depth: Full layer of the octree.
          node_displacement: Output per-node displacement.
          node_feature: Compute per node feature.
          split_label: Compute per node splitting label.
          adaptive: Build adaptive octree.
          adaptive_depth: Starting depth of adaptive octree.
          threshold_distance: The threshold for simplifying octree.
          threshold_normal: The threshold for simplying octree.
          key2xyz: Convert the key to xyz when serialized.
        """
        self.depth = depth
        self.full_depth = full_depth
        self.node_displacement = node_displacement
        self.node_feature = node_feature
        self.split_label = split_label
        self.adaptive = adaptive
        self.adaptive_depth = adaptive_depth
        self.threshold_distance = threshold_distance
        self.threshold_normal = threshold_normal
        self.key2xyz = key2xyz

class OctreeProcessor(DataProcessor):
    """ Reads points files and processes them into octrees"""
    def __init__(self, octree_settings, augmentors=None):
        """ Initialies OctreeProcessor
        Args:
          octree_settings: OctreeSettings object
          augmentors: List of octree augmentors to augment processing.
        """
        self.octree_settings = octree_settings
        super(OctreeProcessor, self).__init__(augmentors)

    def process(self, file_path, aug_index):
        """ Processess points file into octree
        Args:
          file_path: Path to points file
          aug_index: Augmentation index of total augmentations.
        """
        points = Points(file_path)
        octree_info = OctreeInfo()
        octree_info.initialize(
            self.octree_settings.depth,
            self.octree_settings.full_depth,
            self.octree_settings.node_displacement,
            self.octree_settings.node_feature,
            self.octree_settings.split_label,
            self.octree_settings.adaptive,
            self.octree_settings.adaptive_depth,
            self.octree_settings.threshold_distance,
            self.octree_settings.threshold_normal,
            self.octree_settings.key2xyz,
            points)

        for augmentor in self.augmentors:
            augmentor.augment(points, aug_index)

        radius, center = points.get_points_bounds()
        octree_info.set_bbox(radius, center)

        return Octree(octree_info, points)
