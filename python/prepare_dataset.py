from __future__ import print_function

import argparse

from enum import Enum
from modelnet import ModelNet40, ModelNet10
from dataset import DatasetActions, ConversionArguments

class DatasetTypes(Enum):
    """
        Dataset types that can currently be prepared
    """
    ModelNet10 = 1
    ModelNet40 = 2

    def __str__(self):
        return str(self.name)

def main(data_dir,
         dataset_type,
         points_converter,
         octree_converter,
         lmdb_converter,
         starting_action,
         conversion_arguments):
    """
        Prepares archived dataset to test/train on Caffe by outputting LMDB files.

        Args:
            data_dir (str): Directory to output prepared dataset
            dataset_type (DatasetTypes): Dataset type to prepare
            point_converter_path (str): Path to obj/off to points converter executable
            octree_converter_path (str): Path to points to octree converter executable
            lmdb_converter_path (str): Path to octree to lmdb converter executable.
            starting_action (DatasetActions): Starting action to perform on dataset.
            conversion_arguments (ConversionArguments): Arguments specified to convert to octree
    """

    if dataset_type is DatasetTypes.ModelNet10:
        dataset = ModelNet10(data_dir)
    elif dataset_type is DatasetTypes.ModelNet40:
        dataset = ModelNet40(data_dir)
    else:
        raise AttributeError("Dataset type {0} is not supported".format(dataset_type))

    dataset.prepare_set(conversion_arguments,
                        points_converter,
                        octree_converter,
                        lmdb_converter,
                        starting_action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Required Arguments
    parser.add_argument("--datadir",
                        type=str,
                        help="Base folder containing the training, validation and testing data.",
                        required=True)
    parser.add_argument("--dataset",
                        type=lambda dataset: DatasetTypes[dataset],
                        help="Dataset to evaluate",
                        required=True,
                        choices=list(DatasetTypes))
    parser.add_argument("--points_converter_path",
                        type=str,
                        help="Path to executable which converts off/obj to points files",
                        required=True)
    parser.add_argument("--octree_converter_path",
                        type=str,
                        help="Path to executable which converts points to octree files",
                        required=True,)
    parser.add_argument("--lmdb_converter_path",
                        type=str,
                        help="Path to executable which converts octree to lmdb files",
                        required=True)

    #Non Required Arguments
    parser.add_argument("--starting_action",
                        type=lambda action: DatasetActions[action],
                        help="Starting action to perform",
                        required=False,
                        choices=list(DatasetActions))
    parser.add_argument("--depth",
                        type=int,
                        help="Maximum depth of the octree",
                        required=False,
                        default=6)
    parser.add_argument("--full_layer",
                        type=int,
                        help="Layer of octree which is full",
                        required=False,
                        default=2)
    parser.add_argument("--displacement",
                        type=float,
                        help="Offset value for thin shapes",
                        required=False,
                        default=0.55)
    parser.add_argument("--augmentation",
                        type=int,
                        help="Number of model poses converted to octrees",
                        required=False,
                        default=24)
    parser.add_argument("--for_segmentation",
                        help="Whether model is for segmentation or not",
                        required=False,
                        action="store_true")

    parser.set_defaults(for_segmentation=False)

    args = parser.parse_args()

    conversion_arguments = ConversionArguments(args.depth, args.full_layer, args.displacement,
                                               args.augmentation, args.for_segmentation)
    main(args.datadir,
         args.dataset,
         args.points_converter_path,
         args.octree_converter_path,
         args.lmdb_converter_path,
         args.starting_action,
         conversion_arguments)
