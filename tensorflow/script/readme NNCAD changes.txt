The folder contains modified OCNN tensorflow script files:

network_aenet.py This is the caffe voxel labeling network implemented in tensorflow



tfsolver.py: Reads number of tfrecords in the test set.
	Contain a declaration of a global list used in the command to load tensors used in debuging. 
	And the line change to the session run to extract these tensors.
	
	
ocnn.py: Contains change to the loss function replacing > with tf.greater to fix OCNN bug
