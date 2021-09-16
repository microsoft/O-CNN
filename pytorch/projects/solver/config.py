import os
import sys
import shutil
import argparse
from datetime import datetime
from yacs.config import CfgNode as CN

_C = CN()

# SOLVER related parameters
_C.SOLVER = CN()
_C.SOLVER.alias             = ''         # The experiment alias
_C.SOLVER.gpu               = (0,)       # The gpu ids
_C.SOLVER.run               = 'train'    # Choose from train or test

_C.SOLVER.logdir            = 'logs'     # Directory where to write event logs
_C.SOLVER.ckpt              = ''         # Restore weights from checkpoint file
_C.SOLVER.ckpt_num          = 10         # The number of checkpoint kept

_C.SOLVER.type              = 'sgd'      # Choose from sgd or adam
_C.SOLVER.weight_decay      = 0.0005     # The weight decay on model weights
_C.SOLVER.max_epoch         = 300        # Maximum training epoch
_C.SOLVER.eval_epoch        = 1          # Maximum evaluating epoch
_C.SOLVER.test_every_epoch  = 10         # Test model every n training epochs

_C.SOLVER.lr_type           = 'step'     # Learning rate type: step or cos
_C.SOLVER.lr                = 0.1        # Initial learning rate
_C.SOLVER.gamma             = 0.1        # Learning rate step-wise decay
_C.SOLVER.step_size         = (120,60,)  # Learning rate step size.
_C.SOLVER.lr_power          = 0.9        # Used in poly learning rate

_C.SOLVER.dist_url          = 'tcp://localhost:10001'
_C.SOLVER.progress_bar      = True


# DATA related parameters
_C.DATA = CN()
_C.DATA.train = CN()
_C.DATA.train.name          = ''          # The name of the dataset

# For octree building
# If node_dis = True and there are normals, the octree features
# is 4 channels, i.e., the average normals and the 1 channel displacement. 
# If node_dis = True and there are no normals, the feature is also 4 channels,
# i.e., a 3 channel # displacement of average points relative to the center
# points, and the last channel is constant.
_C.DATA.train.depth         = 5           # The octree depth
_C.DATA.train.full_depth    = 2           # The full depth
_C.DATA.train.node_dis      = False       # Save the node displacement
_C.DATA.train.split_label   = False       # Save the split label
_C.DATA.train.adaptive      = False       # Build the adaptive octree
_C.DATA.train.node_feat     = False       # Calculate the node feature

# For normalization
# If radius < 0, then the method will compute a bounding sphere
_C.DATA.train.bsphere       = 'sphere'    # The method uesd to calc the bounding sphere
_C.DATA.train.radius        = -1.         # The radius and center of the bounding sphere
_C.DATA.train.center        = (-1., -1., -1.)

# For transformation
_C.DATA.train.offset        = 0.016       # Used to displace the points when building octree
_C.DATA.train.normal_axis   = ''          # Used to re-orient normal directions

# For data augmentation
_C.DATA.train.disable       = False       # Disable this dataset or not
_C.DATA.train.distort       = False       # Whether to apply data augmentation
_C.DATA.train.scale         = 0.0         # Scale the points
_C.DATA.train.uniform       = False       # Generate uniform scales
_C.DATA.train.jitter        = 0.0         # Jitter the points
_C.DATA.train.interval      = (1, 1, 1)   # Use interval&angle to generate random angle
_C.DATA.train.angle         = (180, 180, 180)

# For data loading
_C.DATA.train.location      = ''          # The data location
_C.DATA.train.filelist      = ''          # The data filelist
_C.DATA.train.batch_size    = 32          # Training data batch size
_C.DATA.train.num_workers   = 8           # Number of workers to load the data
_C.DATA.train.shuffle       = False       # Shuffle the input data
_C.DATA.train.in_memory     = False       # Load the training data into memory


_C.DATA.test = _C.DATA.train.clone()


# MODEL related parameters
_C.MODEL = CN()
_C.MODEL.name               = ''          # The name of the model
_C.MODEL.depth              = 5           # The input octree depth
_C.MODEL.full_depth         = 2           # The input octree full depth layer
_C.MODEL.depth_out          = 5           # The output feature depth
_C.MODEL.channel            = 3           # The input feature channel
_C.MODEL.factor             = 1           # The factor used to widen the network
_C.MODEL.nout               = 40          # The output feature channel
_C.MODEL.resblock_num       = 3           # The resblock number
_C.MODEL.bottleneck         = 4           # The bottleneck factor of one resblock
_C.MODEL.dropout            = (0.0,)      # The dropout ratio
_C.MODEL.upsample           = 'nearest'   # The method used for upsampling
_C.MODEL.interp             = 'linear'    # The interplation method: linear or nearest
_C.MODEL.nempty             = False       # Perform Octree Conv on non-empty octree nodes
_C.MODEL.sync_bn            = False       # Use sync_bn when training the network
_C.MODEL.use_checkpoint     = False       # Use checkpoint to save memory

# loss related parameters
_C.LOSS = CN()
_C.LOSS.num_class           = 40          # The class number for the cross-entropy loss
_C.LOSS.weights             = (1.0, 1.0)  # The weight factors for different losses
_C.LOSS.label_smoothing     = 0.0         # The factor of label smoothing


# backup the commands
_C.SYS = CN()
_C.SYS.cmds              = ''          # Used to backup the commands

FLAGS = _C


def _update_config(FLAGS, args):
  FLAGS.defrost()
  if args.config:
    FLAGS.merge_from_file(args.config)
  if args.opts:
    FLAGS.merge_from_list(args.opts)
  FLAGS.SYS.cmds = ' '.join(sys.argv)
  # update logdir
  alias = FLAGS.SOLVER.alias.lower()
  if 'time' in alias:
    alias = alias.replace('time', datetime.now().strftime('%m%d%H%M')) #%S
  if alias is not '':
    FLAGS.SOLVER.logdir += '_' + alias
  FLAGS.freeze()


def _backup_config(FLAGS, args):
  logdir = FLAGS.SOLVER.logdir
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  # copy the file to logdir
  if args.config:
    shutil.copy2(args.config, logdir)
  # dump all configs
  filename = os.path.join(logdir, 'all_configs.yaml')
  with open(filename, 'w') as fid:
    fid.write(FLAGS.dump())


def _set_env_var(FLAGS):
  gpus = ','.join([str(a) for a in FLAGS.SOLVER.gpu])
  os.environ['CUDA_VISIBLE_DEVICES'] = gpus


def get_config():
  return FLAGS

def parse_args(backup=True):
  parser = argparse.ArgumentParser(description='The configs')
  parser.add_argument('--config', type=str,
                      help='experiment configure file name')
  parser.add_argument('opts', nargs=argparse.REMAINDER,
                      help="Modify config options using the command-line")

  args = parser.parse_args()
  _update_config(FLAGS, args)
  if backup:
    _backup_config(FLAGS, args)
  _set_env_var(FLAGS)
  return FLAGS


if __name__ == '__main__':
  flags = parse_args(backup=False)
  print(flags)
