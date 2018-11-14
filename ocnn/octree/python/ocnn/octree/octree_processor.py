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

class OctreeProcessor(DataProcessor):
    """ Reads points files and processes them into octrees"""
    def __init__(self, octree_settings, augmentor_collection=None):
        """ Initialies OctreeProcessor
        Args:
          octree_settings: OctreeSettings object
          augmentor_collection: AugmentorCollection object.
        """
        self.octree_settings = octree_settings
        super(OctreeProcessor, self).__init__(augmentor_collection)

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

        self.augmentor_collection.augment(points, aug_index)

        radius, center = points.get_points_bounds()
        octree_info.set_bbox(radius, center)

        return Octree(octree_info, points)
