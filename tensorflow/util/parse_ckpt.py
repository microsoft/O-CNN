import argparse
from numpy import prod
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True,
                    help='The path of the ckpt file')
parser.add_argument('--skips', type=str, required=False, nargs='*',
                    help="Skip specific variables")
args = parser.parse_args()
ckpt, skips = args.ckpt, args.skips
if not skips: skips = []

reader = tf.contrib.framework.load_checkpoint(ckpt)
all_vars = tf.contrib.framework.list_variables(ckpt)

total_num = 0
for idx, var in enumerate(all_vars):
  name, shape = var[0], var[1]

  exclude = False
  for s in skips:
    exclude = s in name
    if exclude: break
  if exclude: continue

  shape_str = '; '.join([str(s) for s in shape])
  shape_num = prod(shape)
  print("{:3}, {}, [{}], {}".format(idx, name, shape_str, shape_num))
  total_num += shape_num

print('Total parameters: {}'.format(total_num))
