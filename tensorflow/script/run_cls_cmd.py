import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=False, default='0202_randinit')
parser.add_argument('--gpu', type=int, required=False, default=0)
parser.add_argument('--mode', type=str, required=False, default='randinit')
parser.add_argument('--ckpt', type=str, required=False,
                    default='dataset/midnet_data/mid_d6_o6/model/iter_800000.ckpt')

args = parser.parse_args()
alias = args.alias
gpu = args.gpu
mode = args.mode

factor = 2
dropout = 0.0
batch_size = 32
ckpt = args.ckpt if mode != 'randinit' else '\'\''
module = 'run_cls_finetune.py' if mode != 'randinit' else 'run_cls.py'
script = 'python %s --config configs/cls_hrnet.yaml' % module
data = 'dataset/ModelNet40/m40_y'
ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
muls   = [4,    4,    2,    1,    1,    1,    1]

for i in range(len(ratios)):
  ratio, mul = ratios[i], muls[i]
  max_iter = int(100000 * ratio * mul)
  test_every_iter = int(1000 * ratio * mul)
  step_size1, step_size2 = int(40000 * ratio * mul), int(20000 * ratio * mul)
  # if ratio == 1.00 and finetune: step_size2, max_iter = 10000, 60000

  cmds = [
      script,
      'SOLVER.gpu {},'.format(gpu),
      'SOLVER.logdir logs/m40/{}/cls_{:.2f}'.format(alias, ratio),
      'SOLVER.max_iter {}'.format(max_iter),
      'SOLVER.step_size {},{}'.format(step_size1, step_size2),
      'SOLVER.test_every_iter {}'.format(test_every_iter),
      'SOLVER.ckpt {}'.format(ckpt),
      'DATA.train.location {}_{:.2f}_train_points.tfrecords'.format(data, ratio),
      'DATA.test.location {}_1.00_test_points.tfrecords'.format(data),
      'MODEL.factor {}'.format(factor),
      'MODEL.dropout {},'.format(dropout)]
  if ratio < 0.1 and mode == 'finetune':
    cmds.append('SOLVER.mode finetune_head')
    cmds.append('DATA.train.jitter 0.25')


  cmd = ' '.join(cmds)
  print('\n', cmd, '\n')
  os.system(cmd)
