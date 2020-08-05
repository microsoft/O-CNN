import argparse
from numpy import prod
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='list_var',
                    help='The command to run')
parser.add_argument('--ckpt', type=str, required=True,
                    help='The path of the ckpt file')
parser.add_argument('--skips', type=str, required=False, nargs='*',
                    help="Skip specific variables")
parser.add_argument('--vars', type=str, required=False, nargs='*',
                    help="The variable names")
args = parser.parse_args()
ckpt, skips, varis = args.ckpt, args.skips, args.vars
if not skips: skips = []
if not varis: varis = []

def print_var():
  for v in varis:
    t = tf.train.load_variable(ckpt, v)
    print(v, '\n', t)

def list_var():
  all_vars = tf.train.list_variables(ckpt)
  total_num = 0
  for idx, var in enumerate(all_vars):
    name, shape = var[0], var[1]

    exclude = False
    for s in skips:
      exclude = s in name
      if exclude:
        break
    if exclude:
      continue

    shape_str = '; '.join([str(s) for s in shape])
    shape_num = prod(shape)
    print("{:3}, {}, [{}], {}".format(idx, name, shape_str, shape_num))
    total_num += shape_num

  print('Total parameters: {}'.format(total_num))

if __name__ == '__main__':
  eval(args.run + '()')