import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True, 
                    help='The command to run.')
parser.add_argument('--convert_octree_data', type=str,  required=False, 
                    help='The path of the convert_octree_data')
parser.add_argument('--scanner', type=str,  required=False, 
                    help='The path of the virtual_scanner')
parser.add_argument('--octree', type=str, required=False,
                    help='The path of the octree')

args = parser.parse_args()
cmd  = args.run
virtual_scanner = args.scanner
octree = args.octree
convert_lmdb = args.convert_octree_data
abs_path = os.path.dirname(os.path.realpath(__file__))


def clean_off_file(filename):
  # read the contents of the file
  with open(filename) as f_check:
    file_str = f_check.read()
  # fix the file
  if file_str[0:3] != 'OFF':
    print('Error: not an OFF file: ', filename)
  elif file_str[0:4] != 'OFF\n':
    print('Info: fix an OFF file: ', filename)
    new_str = file_str[0:3] + '\n' + file_str[3:]
    with open(filename, 'w') as f_rewrite:
      f_rewrite.write(new_str)


def m40_get_filelist(root_folder, train=True, suffix='off'):
  filelist, category = [], []
  folders = os.listdir(root_folder)
  assert(len(folders) == 40)
  for idx, folder in enumerate(folders):
    subfolder = 'train' if train else 'test'
    current_folder = os.path.join(root_folder, folder, subfolder)
    filenames = os.listdir(current_folder)
    for filename in filenames:
      if filename.endswith(suffix):
        filelist.append(os.path.join(folder, subfolder, filename))
        category.append(idx)
  return filelist, category


def m40_move_files(src_folder, des_folder, suffix):
  # run virtualscanner
  folders = os.listdir(src_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      # generate points
      curr_src_folder = os.path.join(src_folder, folder, subfolder)
      curr_des_folder = os.path.join(des_folder, folder, subfolder)
      if not os.path.exists(curr_des_folder): os.makedirs(curr_des_folder)
      filenames = os.listdir(curr_src_folder)
      for filename in filenames:
        if filename.endswith('.points'):
          os.rename(os.path.join(curr_src_folder, filename),
                    os.path.join(curr_des_folder, filename))


def m40_convert_mesh_to_points(root_folder='dataset/ModelNet40'):
  root_folder = os.path.join(abs_path, root_folder)
  # Delete 3 files since the virtualscanner can not deal with them
  filelist = ['cone/train/cone_0117.off', 
              'curtain/train/curtain_0066.off', 
              'car/train/car_0021.off.off']
  for filename in filelist:
    filename = os.path.join(root_folder, filename)
    if os.path.exists(filename): 
      os.remove(filename)

  # clean the off files  
  train_list, _ = m40_get_filelist(root_folder, train=True,  suffix='off')
  test_list, _  = m40_get_filelist(root_folder, train=False, suffix='off')
  filelist = train_list + test_list
  # for filename in filelist:
  #   clean_off_file(os.path.join(root_folder, filename))

  # run virtualscanner
  folders = os.listdir(root_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(root_folder, folder, subfolder)
      cmd = '%s %s 14' % (virtual_scanner,  curr_folder)
      print(cmd)
      # os.system(cmd)
      
  # move points
  m40_move_files(root_folder, root_folder + '.points', 'points')


def m40_convert_points_to_octree(root_folder, depth=5, adaptive=0, node_dis=0):
  folders = os.listdir(root_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(root_folder, folder, subfolder)      
      # write filelist to disk
      filenames = os.listdir(curr_folder)
      filelist_name = os.path.join(curr_folder, 'list.txt')
      with open(filelist_name, 'w') as fid:
        for filename in filenames:
          fid.write(os.path.join(curr_folder, filename) + '\n')
      # run octree
      octree_folder = root_folder[:-6] + 'octree.%d' % depth
      if adaptive == 1: octree_folder = octree_folder + '.adaptive'
      output_path = os.path.join(octree_folder, folder, subfolder)
      if not os.path.exists(output_path): os.makedirs(output_path)
      cmd = '%s --filenames %s --output_path %s --depth %d --adaptive %d --node_dis %d --axis z' % \
            (octree, filelist_name, output_path, depth, adaptive, node_dis)
      os.system(cmd)


def m40_generate_ocnn_lmdb(depth=5):
  # generate octree
  root_folder =  os.path.join(abs_path, 'dataset')
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  m40_convert_points_to_octree(points_folder, depth, adaptive=0, node_dis=0)

  # generate lmdb
  octree_folder = os.path.join(root_folder, 'ModelNet40.octree.%d' % depth)
  for folder in ['train', 'test']:
    train = folder == 'train'
    filelist, idx = m40_get_filelist(octree_folder, train=train, suffix='octree')
    filename = os.path.join(root_folder, 'm40_%s_octree_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))
    lmdb_name = os.path.join(root_folder, 'm40_%d_2_12_%s_lmdb' % (depth, folder))
    cmd = '%s --shuffle %s/ %s %s ' % \
          (convert_lmdb, octree_folder, filename, lmdb_name)
    os.system(cmd)


def m40_generate_aocnn_lmdb(depth=5):
  # generate octree
  root_folder =  os.path.join(abs_path, 'dataset')
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  m40_convert_points_to_octree(points_folder, depth, adaptive=1, node_dis=1)

  # generate lmdb
  octree_folder = os.path.join(root_folder, 'ModelNet40.octree.%d.adaptive' % depth)
  for folder in ['train', 'test']:
    train = folder == 'train'
    filelist, idx = m40_get_filelist(octree_folder, train=train, suffix='octree')
    filename = os.path.join(root_folder, 'm40_%s_adaptive_octree_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):      
        fid.write('%s %d\n' % (filelist[i], idx[i]))
    lmdb_name = os.path.join(root_folder, 'm40_%d_adaptive_2_12_%s_lmdb' % (depth, folder))
    cmd = '%s --shuffle %s/ %s %s ' % \
          (convert_lmdb, root_folder, filename, lmdb_name)
    os.system(cmd)


if __name__ == '__main__':
  if cmd == 'm40_convert_mesh_to_points':
    m40_convert_mesh_to_points()
  elif cmd == 'm40_generate_ocnn_lmdb':
    m40_generate_ocnn_lmdb(depth=5)
  elif cmd == 'm40_generate_aocnn_lmdb':
    m40_generate_aocnn_lmdb(depth=5)
  else:
    print('Unsupported command:' + cmd)
