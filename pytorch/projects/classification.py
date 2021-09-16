import ocnn
import torch

from solver import Solver, Dataset, parse_args


class ClsSolver(Solver):
  def get_model(self, flags):
    if flags.name.lower() == 'lenet':
      model = ocnn.LeNet(flags.depth, flags.channel, flags.nout)
    elif flags.name.lower() == 'resnet':
      model = ocnn.ResNet(flags.depth, flags.channel, flags.nout,
                          flags.resblock_num)
    else:
      raise ValueError
    return model

  def get_dataset(self, flags):
    transform = ocnn.TransformCompose(flags)
    dataset = Dataset(flags.location, flags.filelist, transform, in_memory=True)
    return dataset, ocnn.collate_octrees

  def train_step(self, batch):
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    logits = self.model(octree)
    log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
    loss = torch.nn.functional.nll_loss(log_softmax, label)
    return {'train/loss': loss}

  def test_step(self, batch):
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    logits = self.model(octree)
    log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
    loss = torch.nn.functional.nll_loss(log_softmax, label)
    pred = torch.argmax(logits, dim=1)
    accu = pred.eq(label).float().mean()
    return {'test/loss': loss, 'test/accu': accu}


def main(TheSolver):
  FLAGS = parse_args()
  Solver.main(FLAGS, TheSolver)


if __name__ == "__main__":
  main(ClsSolver)
