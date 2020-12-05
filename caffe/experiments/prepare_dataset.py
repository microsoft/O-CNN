import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True, 
                    help='The command to run.')
parser.add_argument('--converter', type=str, required=False, default='convert_octree_data',
                    help='The path of the convert_octree_data')
parser.add_argument('--scanner', type=str,  required=False, 
                    help='The path of the virtual_scanner')
parser.add_argument('--octree', type=str, required=False, default='octree',
                    help='The path of the octree')
parser.add_argument('--simplify_points', type=str, required=False,
                    help='The path of the simplify_points')
parser.add_argument('--depth', type=int, default=5,
                    help='The octree depth')

args = parser.parse_args()
cmd  = args.run
octree = args.octree
converter = args.converter
virtual_scanner = args.scanner
simplify = args.simplify_points
convert_image = 'convert_imageset'
octree_bbox = 'octree_bbox'
octree2mesh = 'octree2mesh'
mesh2points = 'mesh2points'
chamfer_distance = 'chamfer_distance'
abs_path = os.path.dirname(os.path.realpath(__file__))


def aocnn_ae_compute_chamfer():
  octree_folder = os.path.join(abs_path, 'dataset', 'ShapeNetV1.ae_output')
  datalist_folder = os.path.join(abs_path, 'dataset', 'ShapeNetV1.datalist')
  mesh_folder = os.path.join(abs_path, 'dataset', 'ShapeNetV1.ae_output.mesh')
  points_folder = os.path.join(abs_path, 'dataset', 'ShapeNetV1.ae_output.points')
  points_gt_folder = os.path.join(abs_path, 'dataset', 'ShapeNetV1.points')
  if not os.path.exists(mesh_folder): os.makedirs(mesh_folder)
  if not os.path.exists(points_folder): os.makedirs(points_folder)

  # update octree_bbox
  datalist_name = os.path.join(datalist_folder, 'ae_input_output_octree_list.txt')
  filenames = os.listdir(octree_folder)
  with open(datalist_name, 'w') as fid:
    for filename in filenames:
      fid.write(os.path.join(octree_folder, filename) + '\n')
  
  cmds = [octree_bbox, '--filenames', datalist_name]
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)

  # octree2mesh
  datalist_name = os.path.join(datalist_folder, 'ae_output_octree_list.txt')
  filenames = os.listdir(octree_folder)
  with open(datalist_name, 'w') as fid:
    for filename in filenames:
      if filename.endswith('_output.octree'):
        fid.write(os.path.join(octree_folder, filename) + '\n')
  
  cmds = [octree2mesh, '--filenames', datalist_name, '--output_path', mesh_folder]
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)

  # mesh2points
  datalist_name = os.path.join(datalist_folder, 'ae_output_mesh_list.txt')
  filenames = os.listdir(mesh_folder)
  with open(datalist_name, 'w') as fid:
    for filename in filenames:
      if filename.endswith('_output.obj'):
        fid.write(os.path.join(mesh_folder, filename) + '\n')
  
  cmds = [mesh2points, '--filenames', datalist_name,
          '--output_path', points_folder, '--area_unit', '1e-4']
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)

  # chamferdistance
  octreelist_a = os.path.join(datalist_folder, 'octree_test_shuffle.txt')
  with open(octreelist_a, 'r') as fid:
    octree_names = fid.readlines()
  datalist_a = os.path.join(datalist_folder, 'ae_output_points_a.txt')
  with open(datalist_a, 'w') as fid:
    for line in octree_names:
      pos = line.find('_7_2_000.octree')
      fid.write(os.path.join(points_gt_folder, line[:pos] + '.points\n'))

  datalist_b = os.path.join(datalist_folder, 'ae_output_points_b.txt')
  points_names = sorted(os.listdir(points_folder))
  with open(datalist_b, 'w') as fid:
    for line in points_names:
      fid.write(os.path.join(points_folder, line) + '\n')
  
  chamfer_results = os.path.join(abs_path, 'dataset', 'ShapeNetV1.chamfer.csv')
  cmds = [chamfer_distance,
          '--filenames_a', datalist_a, '--filenames_b', datalist_b,
          '--filename_out', chamfer_results]
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)


def shapenet_convert_points_to_octree_ae():
  root_folder = os.path.join(abs_path, 'dataset')
  points_folder = os.path.join(root_folder, 'ShapeNetV1.points')
  octree_folder = os.path.join(root_folder, 'ShapeNetV1.octree')
  datalist_folder = os.path.join(root_folder, 'ShapeNetV1.datalist')
  if not os.path.exists(octree_folder): os.mkdir(octree_folder)
  if not os.path.exists(datalist_folder): os.mkdir(datalist_folder)
  category = ['02691156', '02828884', '02933112', '02958343', '03001627',
              '03211117', '03636649', '03691459', '04090263', '04256520',
              '04379243', '04401088', '04530566']

  for i in range(0, len(category)):
    print('Processing ' + category[i])
    points_category = os.path.join(points_folder, category[i])
    octree_category = os.path.join(octree_folder, category[i])

    # generate the datalist for octree.exe
    filename_points = os.listdir(points_category)
    filename_list = os.path.join(
        datalist_folder, category[i] + '_points_list.txt')
    with open(filename_list, 'w') as f:
      for item in filename_points:
        f.write('%s/%s\n' % (points_category, item))

    # call octree
    cmds = [octree, '--filenames', filename_list, '--output_path', octree_category,
            '--adp_depth', '4', '--depth', '7', '--adaptive', '1',
            '--th_distance', '2', '--th_normal', '0.1',
            '--node_dis', '1', '--split_label', '1']
    cmd = ' '.join(cmds)
    print(cmd)
    os.system(cmd)


def shapenet_lmdb_ae():
  root_folder = os.path.join(abs_path, 'dataset')
  points_folder = os.path.join(root_folder, 'ShapeNetV1.points')
  octree_folder = os.path.join(root_folder, 'ShapeNetV1.octree')
  img_folder = os.path.join(root_folder, 'ShapeNetV1.renderings')
  datalist_folder = os.path.join(root_folder, 'ShapeNetV1.datalist')
  lmdb_folder = os.path.join(root_folder, 'ShapeNetV1.lmdb')
  if not os.path.exists(lmdb_folder): os.mkdir(lmdb_folder)
  category = ['02691156', '02828884', '02933112', '02958343', '03001627',
              '03211117', '03636649', '03691459', '04090263', '04256520',
              '04379243', '04401088', '04530566']
  
  shapenet_convert_points_to_octree_ae()

  # generate datalist for octree
  filename_octree_train = os.path.join(datalist_folder, 'octree_train.txt')
  filename_octree_train_aug = os.path.join(datalist_folder, 'octree_train_aug.txt')
  filename_octree_test = os.path.join(datalist_folder, 'octree_test.txt')
  file_octree_train = open(filename_octree_train, 'w')
  file_octree_train_aug = open(filename_octree_train_aug, 'w')
  file_octree_test = open(filename_octree_test, 'w')

  for i in range(0, len(category)):
    path_img  = os.path.join(img_folder, category[i])
    path_point = os.path.join(points_folder, category[i])
    filename_img = sorted(os.listdir(path_img))
    filename_pc = sorted(os.listdir(path_point))
    filename = [val for val in filename_img if val + '.points' in filename_pc]

    for item in filename[:int(len(filename) * 0.8)]:
      file_octree_train.write('%s/%s_7_2_000.octree %d\n' % (category[i], item, i))
    for item in filename[int(len(filename) * 0.8):]:
      file_octree_test.write('%s/%s_7_2_000.octree %d\n' % (category[i], item, i))
    for item in filename[:int(len(filename) * 0.8)]:
      file_octree_train_aug.write('%s/%s_7_2_000.octree %d\n' % (category[i], item, i))
      file_octree_train_aug.write('%s/%s_7_2_001.octree %d\n' % (category[i], item, i))
      file_octree_train_aug.write('%s/%s_7_2_011.octree %d\n' % (category[i], item, i))

  file_octree_train.close()
  file_octree_test.close()
  file_octree_train_aug.close()


  # generate lmdb for octree
  print('Generate octree lmdb ...')
  cmds = [converter, octree_folder + '/', filename_octree_train, lmdb_folder + '/octree_train_lmdb']
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)

  cmds = [converter, octree_folder + '/', filename_octree_train_aug, lmdb_folder + '/octree_train_aug_lmdb']
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)

  cmds = [converter, octree_folder + '/', filename_octree_test, lmdb_folder + '/octree_test_lmdb']
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)

  # generate datalist for image
  filename_octree_train = os.path.join(datalist_folder, 'octree_train_shuffle.txt')
  filename_img_train = os.path.join(datalist_folder, 'img_train_shuffle.txt')
  file_img_train = open(filename_img_train, 'w')
  file_octree_train = open(filename_octree_train, 'r')  
  lines = file_octree_train.readlines()
  rand_perm = []
  view_num = 24
  for i in range(0, len(lines)):
    rand_perm.append(numpy.random.permutation(view_num))
  for v in range(0, view_num):
    for i in range(0, len(lines)):
      line_img = lines[i].replace('/', '/') \
                         .replace('_7_2_000.octree',
                                  '/rendering/%02d.png' % rand_perm[i][v])
      file_img_train.write(line_img)
  file_octree_train.close()
  file_img_train.close()

  filename_oct_test = os.path.join(datalist_folder, 'octree_test_shuffle.txt')
  filename_img_test = os.path.join(datalist_folder, 'img_test_shuffle.txt')
  file_img_test = open(filename_img_test, 'w')
  file_oct_test = open(filename_oct_test, 'r')
  lines = file_oct_test.readlines()
  for line in lines:
    line_img = line.replace('/', '/').replace('_7_2_000.octree', '/rendering/00.png')
    file_img_test.write(line_img)
  file_img_test.close()
  file_oct_test.close()

  # generate lmdb for image2shape
  print('Generate image lmdb ...')
  cmds = [convert_image, img_folder+'/', filename_img_train, lmdb_folder+'/img_train_lmdb', 
          '--noshuffle', '--check_size']
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)

  cmds = [convert_image, img_folder+'/', filename_img_test, lmdb_folder+'/img_test_lmdb', 
          '--noshuffle', '--check_size']
  cmd = ' '.join(cmds)
  print(cmd)
  os.system(cmd)


def clean_off_file(filename):
  # read the contents of the file
  with open(filename) as fid:
    file_str = fid.read()
  # fix the file
  if file_str[0:3] != 'OFF':
    print('Error: not an OFF file: ' + filename)
  elif file_str[0:4] != 'OFF\n':
    print('Info: fix an OFF file: '  + filename)
    new_str = file_str[0:3] + '\n' + file_str[3:]
    with open(filename, 'w') as f_rewrite:
      f_rewrite.write(new_str)


def m40_get_filelist(root_folder, train=True, suffix='off'):
  filelist, category = [], []
  folders = sorted(os.listdir(root_folder))
  assert(len(folders) == 40)
  for idx, folder in enumerate(folders):
    subfolder = 'train' if train else 'test'
    current_folder = os.path.join(root_folder, folder, subfolder)
    filenames = sorted(os.listdir(current_folder))
    for filename in filenames:
      if filename.endswith(suffix):
        filelist.append(os.path.join(folder, subfolder, filename))
        category.append(idx)
  return filelist, category


def m40_move_files(src_folder, des_folder, suffix):
  folders = os.listdir(src_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
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
  # Delete 3 files since the virtualscanner can not well deal with them
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
  for filename in filelist:
    clean_off_file(os.path.join(root_folder, filename))

  # run virtualscanner
  folders = os.listdir(root_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(root_folder, folder, subfolder)
      cmd = '%s %s 14' % (virtual_scanner,  curr_folder)
      print(cmd)
      os.system(cmd)
      
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
          if filename.endswith('.points'):
            fid.write(os.path.join(curr_folder, filename) + '\n')
      # run octree
      octree_folder = root_folder[:-6] + 'octree.%d' % depth
      if adaptive == 1: octree_folder = octree_folder + '.adaptive'
      output_path = os.path.join(octree_folder, folder, subfolder)
      if not os.path.exists(output_path): os.makedirs(output_path)
      cmd = '%s --filenames %s --output_path %s --depth %d --adaptive %d --node_dis %d --axis z' % \
            (octree, filelist_name, output_path, depth, adaptive, node_dis)
      print(cmd)
      os.system(cmd)


def m40_simplify_points(root_folder='dataset/ModelNet40.points', resolution=64):
  # rename and backup the original folders
  root_folder = os.path.join(abs_path, root_folder)
  original_folder = root_folder + ".dense"
  if os.path.exists(root_folder):
    os.rename(root_folder, original_folder)

  folders = os.listdir(original_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(original_folder, folder, subfolder)      
      # write filelist to disk
      filenames = os.listdir(curr_folder)
      filelist_name = os.path.join(curr_folder, 'list.txt')
      with open(filelist_name, 'w') as fid:
        for filename in filenames:
          if filename.endswith('.points'):
            fid.write(os.path.join(curr_folder, filename) + '\n')
      # run simplify_points
      output_path = os.path.join(root_folder, folder, subfolder)
      if not os.path.exists(output_path): os.makedirs(output_path)
      cmd = '%s --filenames %s --output_path %s --dim %d' % \
            (simplify, filelist_name, output_path, resolution)
      print(cmd)
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
    shuffle = '--shuffle' if train else '--noshuffle'
    filelist, idx = m40_get_filelist(octree_folder, train=train, suffix='octree')
    filename = os.path.join(root_folder, 'm40_%s_octree_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))
    lmdb_name = os.path.join(root_folder, 'm40_%d_2_12_%s_lmdb' % (depth, folder))
    cmd = '%s %s %s/ %s %s ' % \
          (converter, shuffle, octree_folder, filename, lmdb_name)
    print(cmd)
    os.system(cmd)


def m40_generate_aocnn_lmdb(depth=5):
  # generate octree
  root_folder =  os.path.join(abs_path, 'dataset')
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  m40_convert_points_to_octree(points_folder, depth, adaptive=1, node_dis=0)

  # generate lmdb
  octree_folder = os.path.join(root_folder, 'ModelNet40.octree.%d.adaptive' % depth)
  for folder in ['train', 'test']:
    train = folder == 'train'
    shuffle = '--shuffle' if train else '--noshuffle'
    filelist, idx = m40_get_filelist(octree_folder, train=train, suffix='octree')
    filename = os.path.join(root_folder, 'm40_%s_adaptive_octree_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))
    lmdb_name = os.path.join(root_folder, 'm40_%d_adaptive_2_12_%s_lmdb' % (depth, folder))
    cmd = '%s %s %s/ %s %s ' % \
          (converter, shuffle, octree_folder, filename, lmdb_name)
    print(cmd)
    os.system(cmd)


def m40_generate_ocnn_points_tfrecords():
  root_folder  = os.path.join(abs_path, 'dataset')
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  for folder in ['train', 'test']:
    train = folder == 'train'
    shuffle = '--shuffle true' if folder == 'train' else ''
    filelist, idx = m40_get_filelist(points_folder, train=train, suffix='points')
    filename = os.path.join(root_folder, 'm40_%s_points_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):      
        fid.write('%s %d\n' % (filelist[i], idx[i]))
    tfrecords_name = os.path.join(root_folder, 'm40_%s_points.tfrecords' % folder)
    cmd = 'python %s %s --file_dir %s --list_file %s --records_name %s' % \
          (converter, shuffle, points_folder, filename, tfrecords_name)
    print(cmd)
    os.system(cmd)


def m40_generate_ocnn_octree_tfrecords(depth=5):
  # generate octree
  root_folder =  os.path.join(abs_path, 'dataset')
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  m40_convert_points_to_octree(points_folder, depth, adaptive=0, node_dis=0)

  # generate tfrecords
  octree_folder = os.path.join(root_folder, 'ModelNet40.octree.%d' % depth)
  for folder in ['train', 'test']:
    train = folder == 'train'
    shuffle = '--shuffle true' if folder == 'train' else ''
    filelist, idx = m40_get_filelist(octree_folder, train=train, suffix='octree')
    filename = os.path.join(root_folder, 'm40_%s_octree_list.txt' % folder)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))    
    tfname = os.path.join(root_folder, 'm40_%d_2_12_%s_octree.tfrecords' % (depth, folder))
    cmd = 'python %s %s --file_dir %s --list_file %s --records_name %s' % \
          (converter, shuffle, octree_folder, filename, tfname)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
  if cmd == 'm40_convert_mesh_to_points':
    m40_convert_mesh_to_points()
  elif cmd == 'm40_generate_ocnn_lmdb':
    m40_generate_ocnn_lmdb(depth=5)
  elif cmd == 'm40_generate_aocnn_lmdb':
    m40_generate_aocnn_lmdb(depth=args.depth)
  elif cmd == 'm40_generate_ocnn_points_tfrecords':
    m40_generate_ocnn_points_tfrecords()
  elif cmd == 'm40_generate_ocnn_octree_tfrecords':
    m40_generate_ocnn_octree_tfrecords()
  elif cmd == 'm40_simplify_points':
    m40_simplify_points()
  elif cmd == 'shapenet_convert_points_to_octree_ae': 
    shapenet_convert_points_to_octree_ae()
  elif cmd == 'shapenet_lmdb_ae':
    shapenet_lmdb_ae()
  elif cmd == 'aocnn_ae_compute_chamfer':
    aocnn_ae_compute_chamfer()
  else:
    print('Unsupported command:' + cmd)
