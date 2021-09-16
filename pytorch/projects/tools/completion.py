import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--octree', type=str, required=False,
                    default='logs/completion/skip_connections_test')
args = parser.parse_args()

abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data/ocnn_completion')

points2ply, ply2points, octree2pts = 'points2ply', 'ply2points', 'octree2points'


def download_point_clouds():
  # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'https://www.dropbox.com/s/z2x0mw4ai18f855/ocnn_completion.zip?dl=0'
  cmd = 'wget %s -O %s.zip' % (url, root_folder)
  print(cmd)
  os.system(cmd)

  # unzip
  cmd = 'unzip %s.zip -d %s' % (root_folder, root_folder)
  print(cmd)
  os.system(cmd)


def _convert_ply_to_points(prefix='shape'):
  ply_folder = os.path.join(root_folder, prefix + '.ply')
  points_folder = os.path.join(root_folder, prefix + '.points')

  folders = os.listdir(ply_folder)
  for folder in folders:
    curr_folder = os.path.join(ply_folder, folder)

    # write filelist to disk
    filenames = os.listdir(curr_folder)
    filelist_name = os.path.join(curr_folder, 'filelist.txt')
    with open(filelist_name, 'w') as fid:
      for filename in filenames:
        if filename.endswith('.ply'):
          fid.write(os.path.join(curr_folder, filename) + '\n')

    # run points2ply
    output_path = os.path.join(points_folder, folder)
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    cmd = '%s --filenames %s --output_path %s --verbose 0' % \
          (ply2points, filelist_name, output_path)
    print(cmd)
    os.system(cmd)
    os.remove(filelist_name)


def convert_ply_to_points():
  _convert_ply_to_points('shape')
  _convert_ply_to_points('test.scans')


def _convert_points_to_ply(prefix='shape'):
  ply_folder = os.path.join(root_folder, prefix + '.ply')
  points_folder = os.path.join(root_folder, prefix + '.points')

  folders = os.listdir(points_folder)
  for folder in folders:
    curr_folder = os.path.join(points_folder, folder)

    # write filelist to disk
    filenames = os.listdir(curr_folder)
    filelist_name = os.path.join(curr_folder, 'filelist.txt')
    with open(filelist_name, 'w') as fid:
      for filename in filenames:
        if filename.endswith('.points'):
          fid.write(os.path.join(curr_folder, filename) + '\n')

    # run points2ply
    output_path = os.path.join(ply_folder, folder)
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    cmd = '%s --filenames %s --output_path %s --verbose 0' % \
          (points2ply, filelist_name, output_path)
    print(cmd)
    os.system(cmd)
    os.remove(filelist_name)


def convert_points_to_ply():
  _convert_points_to_ply('shape')
  _convert_points_to_ply('test.scans')


def generate_dataset():
  download_point_clouds()
  convert_ply_to_points()


def rename_output_octree():
  filelist = os.path.join(root_folder, 'filelist_test_scans.txt')
  filenames = []
  with open(filelist, 'r') as fid:
    for line in fid:
      filename = line.split()[0]
      filenames.append(filename[:-6] + 'octree')

  idx = 0
  folder_in = args.octree
  octree_in = os.listdir(folder_in)
  octree_in.sort()
  folder_out = os.path.join(root_folder, 'output.octree')
  for o in octree_in:
    if o.endswith('output.octree'):
      name_in = os.path.join(folder_in, o)
      name_out = os.path.join(folder_out, filenames[idx])
      os.renames(name_in, name_out)
      idx += 1
  assert (idx == 1200)


def _convert_octree_to_points(suffix='ply'):
  octree_folder = os.path.join(root_folder, 'output.octree')
  points_folder = os.path.join(root_folder, 'output.' + suffix)

  folders = os.listdir(octree_folder)
  for folder in folders:
    curr_folder = os.path.join(octree_folder, folder)

    # write filelist to disk
    filenames = os.listdir(curr_folder)
    filelist_name = os.path.join(curr_folder, 'filelist.txt')
    with open(filelist_name, 'w') as fid:
      for filename in filenames:
        if filename.endswith('.octree'):
          fid.write(os.path.join(curr_folder, filename) + '\n')

    # run octree2points
    output_path = os.path.join(points_folder, folder)
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    cmd = '%s --filenames %s --output_path %s --verbose 0 --suffix %s' % \
          (octree2pts, filelist_name, output_path, suffix)
    print(cmd)
    os.system(cmd)
    os.remove(filelist_name)


def convert_octree_to_points():
  _convert_octree_to_points('points')
  _convert_octree_to_points('ply')


if __name__ == '__main__':
  eval('%s()' % args.run)
