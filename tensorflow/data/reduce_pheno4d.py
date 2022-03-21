import argparse
import os
import shutil
import re
import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--target', type = int, required = False,
                    help = 'ex. --target 100000',
                    default = 100000)
args = parser.parse_args()

target = args.target

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(current_path, 'script/dataset/pheno4d_segmentation')
dataset_folder = 'ply'
ply_folder = os.path.join(root_folder, dataset_folder)
reduced_folder = os.path.join(root_folder, 'ply_reduced_' + str(target))
cc_header_folder = os.path.join(root_folder, 'cc_header')
cloudcompare = os.path.join(project_root, 'cloudcompare/CloudCompare_v2.12.beta_bin_x64/CloudCompare.exe')
print(cloudcompare)

categories = ['Maize01', 'Maize02', 'Maize03', 'Maize04', 'Maize05', 'Maize06', 'Maize07',
              'Tomato01', 'Tomato02', 'Tomato03', 'Tomato04', 'Tomato05', 'Tomato06', 'Tomato07']

def check_cloudcompare():
  if not(Path(cloudcompare).is_file()):
    print('CloudCompare.exe not found.')
    quit()

def modify_ply_header_cc():
  print('\nModify ply files header ...')
  header = 'ply\nformat ascii 1.0\nelement vertex %d\n' + \
           'property float x\nproperty float y\nproperty float z\n' + \
           'property float nx\nproperty float ny\nproperty float nz\n' + \
           'property float scalar_label\nelement face 0\n' + \
           'property list uchar int vertex_indices\nend_header'
  for i, c in enumerate(categories):
    src_folder = os.path.join(ply_folder, c)
    des_folder = os.path.join(cc_header_folder, c)
    if not os.path.exists(des_folder): os.makedirs(des_folder)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      print(filename)
      filename_ply = os.path.join(src_folder, filename)
      filename_cc = os.path.join(des_folder, filename)
      ply_header = False
      with open(filename_ply, 'r') as fid:
        lines = []
        for line in fid:
          if line == '\n': continue
          nums = line.split()
          if nums[0] == 'end_header':
            ply_header = True
            continue
          if(not(ply_header)): continue
          lines.append(' '.join(nums))

      ply_header = header % len(lines)
      ply_content = '\n'.join([ply_header] + lines)
      with open(filename_cc, 'w') as fid:
        fid.write(ply_content)

def reduce_points():
  print('\nReduce points ...')
  for i, c in enumerate(categories):
    src_folder = os.path.join(cc_header_folder, c)
    des_folder = os.path.join(reduced_folder, c)
    if not os.path.exists(des_folder): os.makedirs(des_folder)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      filename_ply = os.path.join(src_folder, filename)
      filename_reduced_ply = os.path.join(src_folder, filename[:-4] + '_REDUCED.ply')
      filename_reduced = os.path.join(des_folder, filename[:-4] + '.ply')
      cmd = 'python reduce_points.py --cloudcompare %s --file %s --target %s' % (cloudcompare, filename_ply, str(target))
      print(cmd)
      os.system(cmd)
      shutil.move(filename_reduced_ply, filename_reduced)

def modify_ply_header():
  print('\nModify ply files header ...')
  header = 'ply\nformat ascii 1.0\nelement vertex %d\n' + \
           'property float x\nproperty float y\nproperty float z\n' + \
           'property float nx\nproperty float ny\nproperty float nz\n' + \
           'property float label\nelement face 0\n' + \
           'property list uchar int vertex_indices\nend_header'
  for i, c in enumerate(categories):
    src_folder = os.path.join(reduced_folder, c)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      print(filename)
      filename_ply = os.path.join(src_folder, filename)
      ply_header = False
      with open(filename_ply, 'r') as fid:
        lines = []
        for line in fid:
          if line == '\n': continue
          nums = line.split()
          if nums[0] == 'end_header':
            ply_header = True
            continue
          if(not(ply_header)): continue
          lines.append(' '.join(nums))

      ply_header = header % len(lines)
      ply_content = '\n'.join([ply_header] + lines)
      with open(filename_ply, 'w') as fid:
        fid.write(ply_content)

def compute_stats():
  print('\nCompute stats ...')
  max_points = 0
  min_points = 10**9
  avg_points = 0
  clouds = 0
  for i, c in enumerate(categories):
    src_folder = os.path.join(reduced_folder, c)

    filenames = os.listdir(src_folder)
    clouds += len(filenames)
    for filename in filenames:
      filename_ply = os.path.join(src_folder, filename)
      with open(filename_ply, 'r') as fid:
        for line in fid:
          if line == '\n': continue
          nums = line.split()
          if nums[0] == 'element' and nums[1] == 'vertex':
            points = int(nums[2])
            avg_points += points
            max_points = max(max_points, points)
            min_points = min(min_points, points)
            break
  avg_points /= clouds
  ct = datetime.datetime.now()
  ct = str(ct).replace('-', '_').replace(':', '_').replace('.', '_')
  log_content = '\n'.join([
    ct,
    'Point clouds: ' + str(clouds),
    'Max points: ' + str(max_points),
    'Min points: ' + str(min_points),
    'Average points: ' + str(avg_points)])
  print(log_content)
  filename_log = os.path.join(reduced_folder, 'log_' + ct + '.txt')
  with open(filename_log, 'w') as fid:
        fid.write(log_content)

if __name__ == '__main__':
  check_cloudcompare()
  modify_ply_header_cc()
  reduce_points()
  shutil.rmtree(cc_header_folder)
  modify_ply_header()
  compute_stats()
