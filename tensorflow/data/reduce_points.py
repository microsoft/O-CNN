import argparse
import os
import re
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--cloudcompare', type = str, required = True,
                    help = 'ex. --cloudcompare C:\CloudCompare\CloudCompare.exe')
parser.add_argument('--file', type = str, required = True,
                    help = 'ex. --file C:\models\\file.ply')
parser.add_argument('--target', type = int, required = False,
                    help = 'ex. --target 100000',
                    default = 100000)
args = parser.parse_args()

cloudcompare = args.cloudcompare.replace(os.sep, '/')
filename = args.file.replace(os.sep, '/')
current_path = os.path.dirname(filename)
base_filename = Path(filename).stem
target = args.target

# generate 60 (0.01-0.60) subsampled files
for i in range(10,610,10):
  cmd = '%s -silent -c_export_fmt ply -ply_export_fmt ascii -o %s -SS SPATIAL %s' % (cloudcompare, filename, str(i/1000))
  print(cmd)
  os.system(cmd)

# find subsampled files
filenames = os.listdir(current_path)
subsample_files = []
for subsample in filenames:
  filename_subsample = '^' + base_filename + '_SPATIAL_SUBSAMPLED*'
  if re.search(filename_subsample, subsample):
    subsample_files.append(subsample)

# find closest to 100k points from subsampled files
closest_diff = 10**9 # set difference to 1 billion
closest_file = ''
for subsample in subsample_files:
  filename_subsample = os.path.join(current_path, subsample)
  with open(filename_subsample, 'r') as fid:
    for line in fid:
      if line == '\n': continue
      nums = line.split()
      if nums[0] == 'element' and nums[1] == 'vertex':
        diff = abs(target-int(nums[2])) # absolute difference from target
        if diff < closest_diff:
          closest_diff = diff
          closest_file = filename_subsample
        break
      
# save subsample closest to 100k points
filename_reduced = os.path.join(current_path, base_filename + '_REDUCED.ply')
shutil.copy(closest_file, filename_reduced)

# delete generated subsampled files
for subsample in subsample_files:
  filename_subsample = os.path.join(current_path, subsample)
  os.remove(filename_subsample)