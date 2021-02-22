import tensorflow as tf

from config import parse_args, FLAGS
from tfsolver import TFSolver
from ocnn import get_variables_with_name, Optimizer
from run_seg_shapenet import ComputeGraphSeg


class FinetuneOptimizer:
  def __init__(self, flags):
    self.flags = flags.SOLVER

  # build the solver: two different learning rate,
  # the learning rate of backbone is 0.1x smaller
  def __call__(self, total_loss, learning_rate):
    FLAGS = self.flags
    var_list = get_variables_with_name(
        name='ocnn', without='seg_header', verbose=FLAGS.verbose)
    optim_backbone = Optimizer(var_list=var_list, mul=0.1)
    solver1, lr1 = optim_backbone(total_loss, learning_rate)

    var_list = get_variables_with_name(
        name='seg_header', verbose=FLAGS.verbose)
    optim_header = Optimizer(var_list=var_list, mul=1.0)
    solver2, lr2 = optim_header(total_loss, learning_rate)

    solver = tf.group([solver1, solver2])
    return solver, lr2


class FC2Optimizer:
  def __init__(self, flags):
    self.flags = flags.SOLVER

  def __call__(self, total_loss, learning_rate):
    var_list = get_variables_with_name(
        name='seg_header', verbose=self.flags.verbose)
    optim_header = Optimizer(var_list=var_list, mul=1.0)
    solver2, lr2 = optim_header(total_loss, learning_rate)
    return solver2, lr2


class ShapeNetFinetune(TFSolver):
  def __init__(self, flags, compute_graph, build_solver):
    super(ShapeNetFinetune, self).__init__(flags, compute_graph, build_solver)

  def restore(self, sess, ckpt):
    # !!! Restore the trainable/untrainable variables under the name scope `ocnn`
    # Note the variables added by solvers are filtered out since they are not
    # under the scope of `ocnn`
    print('Restore from: ' + ckpt)
    var_restore = get_variables_with_name(
        'ocnn', without='predict_6/conv2', verbose=self.flags.verbose, train_only=False)
    tf_saver = tf.train.Saver(var_list=var_restore)
    tf_saver.restore(sess, ckpt)


# run the experiments
if __name__ == '__main__':
  FLAGS.SOLVER.mode = 'finetune'
  FLAGS = parse_args()
  compute_graph = ComputeGraphSeg(FLAGS)
  optimizer = FinetuneOptimizer if FLAGS.SOLVER.mode == 'finetune' else FC2Optimizer
  build_solver = optimizer(FLAGS)
  solver = ShapeNetFinetune(FLAGS, compute_graph, build_solver)
  solver.run()
