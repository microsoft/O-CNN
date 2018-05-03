import fnmatch
import os

def line_separator_generator(file_list, use_os_sep=True):
    """
        Generates files paths with line separation

        Args:
            file_list (iterable of str): iterable of file paths
            use_os_sep (bool): indicate whether to use os line separator, otherwitse use \n

        Yields:
            str: file path with line separator
    """
    if use_os_sep:
        line_sep = os.linesep
    else:
        line_sep = '\n'

    for file_path in file_list:
        yield file_path + line_sep

def write_out_iterable(output_file, iterable_list):
    """
        Saves file list to file

        Args:
            output_file (str): File to output iterable list to.
            iterable_list (iterable of str): iterable list to write to file
    """
    with open(output_file, 'w') as out_file:
        out_file.writelines(iterable_list)

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
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
            if find_first_instance:
                break
    return matches
