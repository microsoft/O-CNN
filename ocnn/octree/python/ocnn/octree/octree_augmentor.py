"""Module containing OctreeAugmentors"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import yaml

import numpy as np

from ocnn.dataset.augmentor import Augmentor, AugmentorCollection

class OctreeAugmentorCollection(AugmentorCollection):
    """Collection of Augmentors for Octrees"""
    def add_rotation_augmentor(self):
        """Add rotation augmentor"""
        self._augmentors.append(RotationAugmentor(self.num_aug))
    def add_displacing_augmentor(self, displacement, depth):
        """Add displacing augmentor
        Args:
          displacement: The displacement magnitude.
          depth: Total depth of octree to be created.
        """
        self._augmentors.append(DisplacingAugmentor(displacement, depth))
    def add_centering_augmentor(self):
        """Add centering augmentor"""
        self._augmentors.append(CenteringAugmentor())

class RotationAugmentor(Augmentor):
    """Rotates points object to align with points on Fibonacci sphere """
    def __init__(self, total_aug):
        """ Initializes RotationAugmentor
        Args:
          total_aug: Number of points to distribute along Fibonacci sphere.
        """
        self.total_aug = total_aug
        self.rnd = random.random() * total_aug

    def _calculate_fib(self, aug_index):
        """ Calcultes rotation matrix to align to point on Fibonacci sphere
        Args:
          aug_index: Index of point to align to
        """
        offset = 2. / self.total_aug
        increment = np.pi * (3. - np.sqrt(5.))

        y = ((aug_index * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - pow(y,2))

        fib_phi = ((aug_index + self.rnd) % self.total_aug) * increment

        x = np.cos(fib_phi) * r
        z = np.sin(fib_phi) * r

        theta = np.arccos(z)
        phi = np.arctan2(y, x)

        rot_mat = np.array([[np.cos(theta)*np.cos(phi), -np.sin(phi), np.sin(theta)*np.cos(phi)],
                            [np.cos(theta)*np.sin(phi), np.cos(phi), np.sin(theta)*np.sin(phi)],
                            [-np.sin(theta), 0, np.cos(theta)]], np.float32)
        return rot_mat

    def augment(self, item, aug_index):
        """ Rotates points object to align to point on Fibonacci sphere
        Args:
          item: Points object to augment
          aug_index: Index of point to align to
        """
        rot_mat = self._calculate_fib(aug_index)
        item.transform(rot_mat)

class DisplacingAugmentor(Augmentor):
    """Displaces points along their normal direction """
    def __init__(self, displacement, depth):
        """Initializes DisplacingAugmentor
        Args:
          displacement: The displacement magnitude.
          depth: Total depth of octree to be created.
        """
        self.displacement = displacement
        self.depth = depth

    def augment(self, item, aug_index=0):
        """ Displaces points objects points along their normals
        Args:
          item: Points object to augment
        """
        radius, _ = item.get_points_bounds()
        normalized_offset = self.displacement * 2.0 * radius / 2**self.depth
        item.displace(normalized_offset)

class CenteringAugmentor(Augmentor):
    """Moves center of point cloud to origin"""

    def augment(self, item, aug_index=0):
        """Moves center of point cloud to origin
        Args:
          item: Points object to augment
        """
        item.center()
