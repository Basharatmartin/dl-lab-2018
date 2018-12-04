# Author: Basharat Basharat
# Date  : 30-11-2018


import os
import shutil
import tensorflow as tf
import numpy as np

import pickle
import gzip
from utils import *
import matplotlib.pyplot as plt
from tensorboard_evaluation import Evaluation

n_outputs= 4 
image_x  = 96
image_y  = 96
display_step = 3 
training_epochs = 20 
image_shape = [-1, image_x, image_y, 1]
#batch_size = 64 
#learning_rate = 1e-2
output_directory = 'logs'

history_length = 1

##class CNN (object):
##
##	def __init__ (self, x_features, y_labels, mode, params):
def cnn_model (features, labels, mode, params):

	## building CCN network
	#print('\nBuilding the CNN...')

	inputlayer = tf.reshape(features["x"], [-1, 96, 96, 1])

	##conv1 = tf.layers.conv2d (inputs = inputlayer, filters = 32, kernel_size = 5, padding = "valid", activation = tf.nn.relu)
	##print ("Conv1 shape : ", conv1.shape)
	##pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=2, strides=2)
	##print ("Pool1 shape : ", pool1.shape)
	##conv2 = tf.layers.conv2d (inputs = pool1, filters = 32,	kernel_size = 5, padding = "valid", activation = tf.nn.relu)
	##print ("Conv2 shape : ", conv2.shape)
	##pool2 = tf.layers.max_pooling2d (inputs = conv2, pool_size=2, strides=2)
	##print ("Pool2 shape : ", pool2.shape)

	###flat_size = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]

	##pool2_flat = tf.reshape(pool2, [-1, 21*21*32])
	##print ("Pool2_flat shape : ", pool2_flat.shape)
	##dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
	##print ("Dense shape : ", dense.shape)
	##dropout = tf.layers.dropout (inputs=dense, rate=0.7)
	##print ("Dropout shape : ", dropout.shape)
	##logits = tf.layers.dense(inputs=dense, units=4)
	##print ("Logits shape : ", logits.shape)



	conv1 = tf.layers.conv2d (inputs = inputlayer, filters = 16, kernel_size = 5, strides=2, padding = "valid", activation = tf.nn.relu)
	conv2 = tf.layers.conv2d (inputs = conv1, filters = 32, kernel_size = 5, strides=2, padding = "valid", activation = tf.nn.relu)
	conv3 = tf.layers.conv2d (inputs = conv2, filters = 32, kernel_size = 3, strides=1, padding = "valid", activation = tf.nn.relu)
	
	print ("Conv3 shape : ", conv3.shape)
	conv3_flat = tf.reshape(conv3, [-1, 19*19*32])
	dense1 = tf.layers.dense(inputs=conv3_flat, units=128, activation=tf.nn.relu)
	logits = tf.layers.dense(inputs=dense1, units=4, activation=None)
	 

	# LSTM layer
	##a_lstm = tf.nn.rnn_cell.LSTMCell(num_units=256)
	##a_lstm = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=0.8)
	##a_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[a_lstm])

	##a_init_state = a_lstm.zero_state(batch_size=16, dtype=tf.float32)
	##lstm_in = tf.expand_dims(self.output1, axis=1)

	##a_outputs, a_final_state = tf.nn.dynamic_rnn(cell=a_lstm, inputs=lstm_in, initial_state=a_init_state)
	##a_cell_out = tf.reshape(a_outputs, [-1, 256], name='flatten_lstm_outputs')
	##print ("a_cell_out shape : ", a_cell_out)

	# output layer:
	#self.output = tf.contrib.layers.fully_connected(a_cell_out, 4, activation_fn=None)
	# TODO: Loss and optimizer
	#self.cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.output, labels=self.y_))
	#self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
	#self.cost = tf.square (self.y_ - self.output) 
	#self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

	# TODO: Start tensorflow session
	#self.sess = tf.Session()
	##self.saver = tf.train.Saver()
	##init = tf.global_variables_initializer()
	##self.sess.run (init)
	print('logits shape :: ', logits.shape)
	
	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		#"classes": tf.argmax(input=logits),
		"classes": logits,
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		#"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}


	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	#loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	#loss = tf.losses.softmax_cross_entropy(logits, labels)

	loss = tf.losses.huber_loss(labels=labels, predictions=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])
		#optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	##eval_metric_ops = {
	##	"accuracy": tf.metrics.accuracy(
	##		labels=labels, predictions=predictions["classes"])}
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=logits)}
		
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
		


class Model (object):

	def __init__ (self, params, id, model_dir='./model'):

		self.model = tf.estimator.Estimator(model_fn=cnn_model, model_dir="{}/model{}".format(model_dir, id), params=params)
		#self.model = tf.estimator.Estimator(model_fn=cnn_model, params=params)

	
	def training (self, X_train, y_train, X_valid, y_valid, epochs, batch_size):

		tf.reset_default_graph()

		print('\nTraining phase initiated...\n')

		model_dir = './model'
		tensorboard_dir = './tensorboard'
		tensorboard_eval = Evaluation (tensorboard_dir)


		total_batch_num = int (X_train.shape[0] // batch_size)
		total_batch_num_valid = int (X_valid.shape[0] // batch_size)

		print ("total_batch_num  :: ", total_batch_num)
		print ("batch_size   :: ", batch_size)


		for epoch in range(epochs):
			print ("epoch:  %i  :: " %epoch)

			train_input_fn = tf.estimator.inputs.numpy_input_fn(
				x = {"x" : X_train },
				y = y_train, 
				batch_size = batch_size,
				num_epochs=1,
				shuffle=False)

			self.model.train(input_fn=train_input_fn, steps=100)

			eval_input_fn = tf.estimator.inputs.numpy_input_fn(
				x = {"x" : X_valid },
				y = y_valid,
				batch_size = batch_size,
				num_epochs=1,
				shuffle=False)

			stats = self.model.evaluate(eval_input_fn)

			print (stats)

	def select_action (self, state):

		predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":state}, shuffle=False)
		a = list(self.model.predict(predict_input_fn))[0]['classes']
		a[1] = np.round(a[1])
		a[2] = np.round(a[2] * 3) / 5
		
		return a


	def load(self, file_name):
		self.saver.restore(self.sess, file_name)

	def save(self, file_name):
		self.saver.save(self.sess, file_name)

