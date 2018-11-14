""" Module to manage OctreeSettings """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import yaml

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

    @classmethod
    def from_dict(cls, dictionary):
        """ Creates OctreeSettings from Dictionary
        Args:
          dictionary: dictionary with keys as member variables.
        """
        return cls(
            depth=dictionary['depth'],
            full_depth=dictionary['full_depth'],
            node_displacement=dictionary['node_displacement'],
            node_feature=dictionary['node_feature'],
            split_label=dictionary['split_label'],
            adaptive=dictionary['adaptive'],
            adaptive_depth=dictionary['adaptive_depth'],
            threshold_distance=dictionary['threshold_distance'],
            threshold_normal=dictionary['threshold_normal'],
            key2xyz=dictionary['key2xyz'])

    @classmethod
    def from_yaml(cls, yml_filepath):
        """ Creates OctreeSettings from YAML
        Args:
          yml_filepath: Path to yml config file.
        """

        with open(yml_filepath, 'r') as yml_file:
            config = yaml.load(yml_file)

        #Octree Settings
        parameters = config['octree_settings']
        return cls.from_dict(parameters)

    def write_yaml(self, yml_filepath):
        """ Writes ScannerSettings to YAML
        Args:
          yml_filepath: Filepath to output settings
        """
        data = {'octree_settings': {
            'depth': self.depth,
            'full_depth': self.full_depth,
            'node_displacement': self.node_displacement,
            'node_feature': self.node_feature,
            'split_label': self.split_label,
            'adaptive': self.adaptive,
            'adaptive_depth': self.adaptive_depth,
            'threshold_distance': self.threshold_distance,
            'threshold_normal': self.threshold_normal,
            'key2xyz': self.key2xyz}}

        with open(yml_filepath, 'w') as yml_file:
            yaml.dump(data, yml_file, default_flow_style=False)

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return (self.depth == other.depth and
                    self.full_depth == other.full_depth and
                    self.node_displacement == other.node_displacement and
                    self.node_feature == other.node_feature and
                    self.split_label == other.split_label and
                    self.adaptive == other.adaptive and
                    self.adaptive_depth == other.adaptive_depth and
                    np.isclose(self.threshold_distance, other.treshold_distance) and
                    np.isclose(self.threshold_normal, other.threshold_normal) and
                    self.key2xyz == other.key2xyz)
        return NotImplemented

    def __ne__(self, other):
        return not self == other
