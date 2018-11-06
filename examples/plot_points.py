""" Opens a points file and plots different orientations """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from ocnn.octree import Points
from ocnn.octree import RotationAugmentor, CenteringAugmentor

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt


def plot_points(points_path, total_aug):
    """ Plots points file in different orientations
    Args:
      points_path: Path to points file
      total_aug: Number of orientations to view model
    """
    points = Points(points_path)
    rot = RotationAugmentor(total_aug)
    centerer = CenteringAugmentor()
    centerer.augment(points)

    for i in range(total_aug):
        print(i)

        rot.augment(points, i)
        pts, _ = points.get_points_data()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = pts[:,0]
        y = pts[:,1]
        z = pts[:,2]

        ax.scatter(x, y, z, c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_aspect('equal')

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--pointspath",
                        "-p",
                        type=str,
                        help="Path to points file.",
                        required=True)

    parser.add_argument("--numaug",
                        "-n",
                        type=str,
                        help="Number of view points to plot points file",
                        required=False,
                        default=1)

    args = parser.parse_args()

    plot_points(args.pointspath, args.numaug)
