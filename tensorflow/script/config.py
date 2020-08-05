import os
import sys
import shutil
import argparse
from yacs.config import CfgNode as CN

_C = CN()

# SOLVER related parameters
_C.SOLVER = CN()
_C.SOLVER.gpu             = (0,)      # The gpu ids
_C.SOLVER.logdir          = 'logs'    # Directory where to write event logs
_C.SOLVER.ckpt            = ''        # Restore weights from checkpoint file
_C.SOLVER.run             = 'train'   # Choose from train or test
_C.SOLVER.type            = 'sgd'     # Choose from sgd or adam
_C.SOLVER.max_iter        = 160000    # Maximum training iterations
_C.SOLVER.test_iter       = 100       # Test steps in testing phase
_C.SOLVER.test_every_iter = 1000      # Test model every n training steps
_C.SOLVER.lr_type         = 'step'    # Learning rate type: step or cos
_C.SOLVER.learning_rate   = 0.1       # Initial learning rate
_C.SOLVER.gamma           = 0.1       # Learning rate step-wise decay
_C.SOLVER.step_size       = (40000,)  # Learning rate step size.
_C.SOLVER.ckpt_num        = 100       # The number of checkpoint kept 
_C.SOLVER.var_name        = ('_name',)# Variable names used for finetuning
_C.SOLVER.ignore_var_name = ('_name',)# Ignore variable names when loading ckpt
_C.SOLVER.verbose         = False     # Whether to output some messages


# DATA related parameters
_C.DATA = CN()
_C.DATA.train = CN()
_C.DATA.train.dtype      = 'points'   # The data type: points or octree
_C.DATA.train.x_alias    = 'data'     # The alias of the data
_C.DATA.train.y_alias    = 'label'    # The alias of the target

_C.DATA.train.depth      = 5          # The octree depth
_C.DATA.train.full_depth = 2          # The full depth
_C.DATA.train.node_dis   = False      # Save the node displacement
_C.DATA.train.split_label= False      # Save the split label
_C.DATA.train.adaptive   = False      # Build the adaptive octree
_C.DATA.train.node_feat  = False      # Calculate the node feature

_C.DATA.train.distort    = False      # Whether to apply data augmentation
_C.DATA.train.offset     = 0.55       # Offset used to displace the points
_C.DATA.train.axis       = 'y'        # Rotation axis for data augmentation
_C.DATA.train.scale      = 0.0        # Scale the points
_C.DATA.train.uniform    = False      # Generate uniform scales
_C.DATA.train.jitter     = 0.0        # Jitter the points
_C.DATA.train.drop_dim   = (8, 32)    # The value used to dropout points
_C.DATA.train.dropout    = (0, 0)     # The dropout ratio
_C.DATA.train.stddev     = (0, 0, 0)  # The standard deviation of the random noise
_C.DATA.train.interval   = (1, 1, 1)  # Use interval&angle to generate random angle
_C.DATA.train.angle      = (180, 180, 180)

_C.DATA.train.location   = ''         # The data location
_C.DATA.train.shuffle    = 1000       # The shuffle size
_C.DATA.train.take       = -1         # Use at most `take` elements from this dataset
_C.DATA.train.batch_size = 32         # Training data batch size
_C.DATA.train.mask_ratio = 0.0        # Mask out some point features

# _C.DATA.train.return_iter= False      # Return the data iterator
_C.DATA.train.return_pts = False      # Also return points


_C.DATA.test = _C.DATA.train.clone()


# MODEL related parameters
_C.MODEL = CN()
_C.MODEL.name            = ''         # The name of the model
_C.MODEL.depth           = 5          # The input octree depth
_C.MODEL.depth_out       = 5          # The output feature depth
_C.MODEL.channel         = 3          # The input feature channel
_C.MODEL.factor          = 1          # The factor used to widen the network
_C.MODEL.nout            = 40         # The output feature channel
_C.MODEL.nouts           = 40,        # The output feature channels
_C.MODEL.resblock_num    = 3          # The resblock number
_C.MODEL.bottleneck      = 4          # The bottleneck factor of one resblock
_C.MODEL.dropout         = (0.0,)     # The dropout ratio
_C.MODEL.signal_abs      = False      # Use the absolute value of signal
_C.MODEL.upsample        = 'nearest'  # The method used for upsampling


# loss related parameters
_C.LOSS = CN()
_C.LOSS.num_class        = 40         # The class number for the cross-entropy loss
_C.LOSS.weight_decay     = 0.0005     # The weight decay on model weights
_C.LOSS.sigma            = 0.1        # Use for MID training
_C.LOSS.momentum         = 0.5        # Use for MID training
_C.LOSS.inst_num         = 57449      # The object number in MID training
_C.LOSS.seg_num          = 100        # The clustering number in MID training
_C.LOSS.weights          = (1.0, 1.0) # The weight factors for different losses
_C.LOSS.label_smoothing  = 0.0        # The factor of label smoothing


# backup the commands
_C.SYS = CN()
_C.SYS.cmds              = ''         # Used to backup the commands 

FLAGS = _C


def _update_config(FLAGS, args):
  FLAGS.defrost()
  if args.config:
    FLAGS.merge_from_file(args.config)
  if args.opts:
    FLAGS.merge_from_list(args.opts)
  FLAGS.SYS.cmds = ' '.join(sys.argv)
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # export TF_CPP_MIN_LOG_LEVEL=3

def parse_args(backup=True):
  parser = argparse.ArgumentParser(description='The configs')
  parser.add_argument('--config',
                      help='experiment configure file name',
                      type=str)
  parser.add_argument('opts',
                      help="Modify config options using the command-line",
                      nargs=argparse.REMAINDER)
  args = parser.parse_args()
  _update_config(FLAGS, args)
  if backup: _backup_config(FLAGS, args)
  _set_env_var(FLAGS)
  return FLAGS


if __name__ == '__main__':
  flags = parse_args(backup=False)
  print(flags)

