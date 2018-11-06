""" Generates a dataset with rotationally augmented octrees from points files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import time

from ocnn.caffe import LmdbBuilder
from ocnn.dataset import CsvMappedStructure, FolderMappedStructure
from ocnn.dataset import Dataset
from ocnn.octree import OctreeProcessor, OctreeSettings
from ocnn.octree import RotationAugmentor, CenteringAugmentor, DisplacingAugmentor

FILE_PATTERNS = ['*.points']

def generate_set(num_threads, num_aug, model_folder, db_folder, class_map_path=''):
    """ Generates a dataset with rotationally augmented octrees from points
    files.

    Args:
      num_threads: Number of worker threads.
      num_aug: Number of rotationally augmentation per points file.
      model_folder: Base folder containing the training, validation and testing
        data.  Base folder must be organized like ShapeNet or ModelNet.
      db_folder: Output folder of database.
      class_map_path: Path of CSV Class Map file. Provide if dataset is
        structured like ShapeNet. Otherwise dataset is assumed to be structured
        like ModelNet..
    """
    t_start = time.time()

    builder = LmdbBuilder()
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    if class_map_path:
        structure = CsvMappedStructure(
            base_folder=model_folder,
            patterns=FILE_PATTERNS,
            class_map_path=class_map_path)
    else:
        structure = FolderMappedStructure(
            base_folder=model_folder,
            patterns=FILE_PATTERNS)

    octree_settings = OctreeSettings()
    augmentors = [CenteringAugmentor(), DisplacingAugmentor(0.55, octree_settings.depth)]
    if num_aug > 1:
        augmentors.append(RotationAugmentor(num_aug))
    processor = OctreeProcessor(octree_settings, augmentors=augmentors)

    dataset = Dataset(dataset_structure=structure,
                      data_processor=processor,
                      builder=builder,
                      output_folder=db_folder,
                      num_aug=num_aug,
                      num_threads=num_threads)

    dataset.produce_dataset()

    t_end = time.time()
    print('Total time: ' + str(t_end - t_start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--datadir",
                        "-d",
                        type=str,
                        help="""Base folder containing the training, validation and testing data.
                        Base folder must be organized like ShapeNet or ModelNet""",
                        required=True)

    parser.add_argument("--outputdir",
                        "-o",
                        type=str,
                        help="Folder where database will be output",
                        required=True)

    parser.add_argument("--mappath",
                        "-m",
                        type=str,
                        help="""Path of CSV Class Map file. Assumed dataset is structured like ShapeNet.
                                If value not given assumed dataset is structured like ModelNet.""",
                        required=False,
                        default='')

    parser.add_argument("--augnum",
                        "-a",
                        type=int,
                        help="Number of rotation augmentations to points file",
                        required=False,
                        default=1)

    parser.add_argument("--threadnum",
                        "-t",
                        type=int,
                        help="Number of threads to use",
                        required=False,
                        default=1)

    args = parser.parse_args()

    generate_set(
        args.threadnum,
        args.augnum,
        args.datadir,
        args.outputdir,
        args.mappath)
