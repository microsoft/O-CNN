import argparse
import tensorflow as tf

# Use the following command to inspect the tags contained in the logdir
#     tensorboard --inspect --logdir DIRECTORY_PATH
# Then run this script by providing the --event and --tag


parser = argparse.ArgumentParser()
parser.add_argument('--event', type=str, required=True,
                    help='The path of the event file')
parser.add_argument('--tag', type=str, required=True,
                    help='The tag of value')
args = parser.parse_args()


for e in tf.train.summary_iterator(args.event):
  has_value = False
  msg = '{}'.format(e.step)
  for v in e.summary.value:
    if args.tag in v.tag:
      msg = msg + ', {}'.format(v.simple_value)
      has_value = True
  if has_value: print(msg)

