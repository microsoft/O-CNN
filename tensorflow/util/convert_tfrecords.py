import os
import argparse
import tensorflow as tf
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--file_dir', type=str, required=True, 
                    help='Base folder containing the data')
parser.add_argument('--list_file', type=str, required=True,
                    help='File containing the list of data')
parser.add_argument('--records_name', type=str, required=True,
                    help='Name of tfrecords')
parser.add_argument('--file_type', type=str, required=False, default='data',
                    help='File type')
parser.add_argument('--shuffle', type=str, required=False, default='',
                    help='Whether to shuffle the data order')
shuffle_data = False


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_octree(file):
  with open(file, 'rb') as f:
    octree_bytes = f.read()
  return octree_bytes


def write_data_to_tfrecords(file_dir, list_file, records_name, file_type):
  [data, label, index] = get_data_label_pair(list_file)
  
  writer = tf.python_io.TFRecordWriter(records_name)
  for i in range(len(data)):
    if not i % 1000:
      print('data loaded: {}/{}'.format(i, len(data)))

    octree_file = load_octree(os.path.join(file_dir, data[i]))
    feature = {file_type: _bytes_feature(octree_file),
               'label': _int64_feature(label[i]), 
               'index': _int64_feature(index[i]),
               'filename': _bytes_feature(('%06d_%s' % (i, data[i])).encode('utf8'))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
  writer.close()


def get_data_label_pair(list_file):
  file_list = []
  label_list = []
  with open(list_file) as f:
    for line in f:
      file, label = line.split()
      file_list.append(file)
      label_list.append(int(label))
  index_list = list(range(len(label_list)))

  if shuffle_data:
    c = list(zip(file_list, label_list, index_list))
    shuffle(c)
    file_list, label_list, index_list = zip(*c)
    with open(list_file + '.shuffle.txt', 'w') as f:
      for item in c:
        f.write('{} {}\n'.format(item[0], item[1]))
  return file_list, label_list, index_list



if __name__ == '__main__':
  args = parser.parse_args()
  shuffle_data = args.shuffle
  write_data_to_tfrecords(args.file_dir, 
                          args.list_file, 
                          args.records_name, 
                          args.file_type)
