"""Functions to manipulate files/folders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os

def ensure_parent_directory(file_path):
    """ Ensures parent directory of file_path exists
    Args:
      file_path: Path of file
    """

    parent_directory = os.path.dirname(file_path)
    if not os.path.exists(parent_directory):
        os.mkdir(parent_directory)

def dump_dictionary(dictionary, output_path):
    """ Dumps dictionary to file (non-recursive)
    Args:
      dictionary: dictionary to dump
      output_path: path to dump dictionary
    """
    ensure_parent_directory(output_path)
    with open(output_path, 'w') as f:
        for item in dictionary.items():
            f.write('{0}\n'.format(str(item)))

def find_files(directory_name, pattern, find_first_instance=False):
    """
        Returns iterable filename list based on matching pattern

        Args:
            directory_name (str): Directory to search for files
            pattern (str): Pattern to search for.
            find_first_instance (bool): Whether to early return if pattern found.
    """
    matches = []

    for root, _, filenames in os.walk(directory_name):
        file_paths = (os.path.join(root, filename) for filename in filenames)
        for file_path in fnmatch.filter(file_paths, pattern):
            matches.append(file_path)
            if find_first_instance:
                break

    return matches
