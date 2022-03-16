from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import os

checkpoint_path_default = '/home/ervin/Desktop/Thesis/O-CNN/tensorflow/script/dataset/midnet_data/mid_d6_o6/model/iter_800000.ckpt'
checkpoint_path_custom = '/home/ervin/Desktop/Thesis/O-CNN/tensorflow/script/logs/hrnet/0814_hrnet_d6_o6/model/iter_000002.ckpt'

def inspect():
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path_custom)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
        print("tensor_name: ", key)

# 2 root error(s) found.
#   (0) Not found: Key ocnn_hrnet/front/depth_5/conv_bn_relu/batch_normalization/beta/Momentum not found in checkpoint
#          [[node save_1/RestoreV2 (defined at run_seg_pheno4d_finetune.py:54) ]]
#          [[save_1/RestoreV2/_229]]
#   (1) Not found: Key ocnn_hrnet/front/depth_5/conv_bn_relu/batch_normalization/beta/Momentum not found in checkpoint
#          [[node save_1/RestoreV2 (defined at run_seg_pheno4d_finetune.py:54) ]]
# 0 successful operations.
# 0 derived errors ignored.

def fix(checkpoint_dir, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            # Set the new name
            new_name:str = var_name
            if(new_name != 'solver/global_step' and 'solver/' in new_name):
                new_name = new_name.replace('solver/','')
            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)
        if not dry_run:
            print("Saving now...")
            # Save the variables
            saver = tf.compat.v1.train.Saver()
            print("Save initialized")
            sess.run(tf.compat.v1.global_variables_initializer())
            print("Session ran")
            saver.save(sess, checkpoint_dir)
            print("Saved")

#fix(checkpoint_path_custom,False)
inspect()