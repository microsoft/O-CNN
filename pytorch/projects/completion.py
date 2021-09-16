import os
import ocnn
import torch

from solver import Solver, parse_args, get_config
from datasets import get_completion_dataset


class CompletionSolver(Solver):
  def get_model(self, flags):
    return ocnn.OUNet(flags.depth, flags.channel, flags.nout, flags.full_depth)

  def get_dataset(self, flags):
    return get_completion_dataset(flags)

  def model_forward(self, batch):
    octree_in, octree_gt = batch['octree_in'].cuda(), batch['octree'].cuda()
    output = self.model(octree_in, octree_gt, run='compute_loss')
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output

  def eval_step(self, batch):
    octree_in = batch['octree_in'].cuda()
    octree_out = self.model(octree_in, run='decode_shape')

    iter_num = batch['iter_num']
    filename = os.path.join(self.logdir, '%04d.input.octree' % iter_num)
    octree_in.cpu().numpy().tofile(filename)
    filename = os.path.join(self.logdir, '%04d.output.octree' % iter_num)
    octree_out.cpu().numpy().tofile(filename)


def main(TheSolver):
  get_config().DATA.train.camera_path = '_'  # used to generate partial scans
  get_config().DATA.test.camera_path = '_'
  get_config().DATA.train.scan = True
  get_config().DATA.test.scan = True
  get_config().MODEL.skip_connections = True
  get_config().MODEL.full_depth = 2

  FLAGS = parse_args()
  Solver.main(FLAGS, TheSolver)


if __name__ == '__main__':
  main(CompletionSolver)
