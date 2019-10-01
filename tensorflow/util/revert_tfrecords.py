import os
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--records_name', type=str, required=True,
                    help='Name of tfrecords')
parser.add_argument('--output_path', type=str, required=True,
                    help="Folder to store the data")
parser.add_argument('--list_file', type=str, required=False, default='filelist.txt',
                    help='File to record the list of the data')
parser.add_argument('--count', type=int, required=False, default=0,
                    help='Specify the count of the data to extract')
parser.add_argument('--file_type', type=str, required=False, default='data',
                    help='File type')


def read_data_from_tfrecords(records_name, output_path, list_file, file_type, count):
  records_iterator = tf.python_io.tf_record_iterator(records_name)
  count = count if count != 0 else float('Inf')

  with open(os.path.join(output_path, list_file), "w") as f:
    num = 0
    for string_record in records_iterator:
      if num >= count: break

      example = tf.train.Example()
      example.ParseFromString(string_record)
      label  = int(example.features.feature['label'].int64_list.value[0])
      # index  = int(example.features.feature['index'].int64_list.value[0]) 
      octree = example.features.feature[file_type].bytes_list.value[0]
      if 'filename' in example.features.feature:
        filename = example.features.feature['filename'].bytes_list.value[0] \
                          .decode('utf8').replace('/', '_').replace('\\', '_')
      else:
        filename = '%06d.%s' % (num, file_type)

      num += 1
      with open(os.path.join(output_path, filename), 'wb') as fo:
        fo.write(octree)

      f.write("{} {}\n".format(filename, label))


if __name__ == '__main__':
  args = parser.parse_args()

  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  read_data_from_tfrecords(args.records_name,
                           args.output_path,
                           args.list_file,
                           args.file_type,
                           args.count)
