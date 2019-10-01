import os
import tensorflow as tf

tf_flags = tf.app.flags

tf_flags.DEFINE_string('logdir', '', 'Directory where to write event logs.')
tf_flags.DEFINE_string('run', 'train', 'Choose from train or test}.')
tf_flags.DEFINE_string('train_data','', 'Training data location.')
tf_flags.DEFINE_string('test_data','', 'Testing data location.')
tf_flags.DEFINE_integer('train_batch_size', 32, 'Batch size for the training.')
tf_flags.DEFINE_integer('test_batch_size', 32, 'Batch size for the training.')
tf_flags.DEFINE_integer('test_iter', 100, 'Test steps in testing phase.')
tf_flags.DEFINE_integer('max_iter', 160000, 'Maximum training iterations.')
tf_flags.DEFINE_integer('test_every_iter', 1000, 'Test model every n training steps.')
tf_flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
tf_flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on model weights.')
tf_flags.DEFINE_float('gamma', 0.1, 'SGD lr step gamma.')
tf_flags.DEFINE_string('ckpt', '', 'Restore weights from checkpoint file.')
tf_flags.DEFINE_string('gpu', '0', 'The gpu number.')
tf_flags.DEFINE_integer('num_class', 40, 'The class number.')
tf_flags.DEFINE_integer('depth', 5, 'The octree depth.')
tf_flags.DEFINE_integer('channel', 3, 'The input feature channel.')
tf_flags.DEFINE_integer('padding_size', 2, 'The input feature padding size.')
tf_flags.DEFINE_integer('res_block_num', 3, 'The resblock number.')
tf_flags.DEFINE_multi_integer('step_size', None, 'SGD lr step size.')



FLAGS = tf_flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def learning_rate(global_step):
  # lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
  #     FLAGS.step_size, FLAGS.gamma, staircase=True)

  step_size = FLAGS.step_size
  for i in range(len(step_size), 5): 
    step_size.append(step_size[-1])

  steps = step_size
  for i in range(1, 5):
    steps[i] = steps[i-1] + steps[i]
  lr_values = [FLAGS.gamma**i * FLAGS.learning_rate for i in range(0, 6)]

  lr = tf.train.piecewise_constant(global_step, steps, lr_values)
  return lr