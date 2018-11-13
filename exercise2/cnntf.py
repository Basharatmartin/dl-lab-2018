### Author : Basharat Basharat
### Email  : basharat.basharat@uranus.uni-freiburg.de

from __future__ import print_function

import tensorflow as tf
import numpy as np

class CnnTensorflow():

	def model(features, labels, mode):
		
		inputlayer = tf.reshape(features["x"], [-1, 28, 28, 1])

		conv1 = tf.layers.conv2d (
					inputs = inputlayer,
					filters = 16,
					kernel_size = 5,
					padding = "same",
					activation = tf.nn.relu)
		
		print ("Conv1 shape : ", conv1.shape)
		pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=2, strides=1)
		print ("Pool1 shape : ", pool1.shape)

		conv2 = tf.layers.conv2d (
					inputs = pool1, 
					filters = 16,
					kernel_size = 5,
					padding = "same",
					activation = tf.nn.relu)
		print ("Conv2 shape : ", conv2.shape)

		pool2 = tf.layers.max_pooling2d (inputs = conv2, pool_size=2, strides=1)
		print ("Pool2 shape : ", pool2.shape)

		pool2_flat = tf.reshape(pool2, [-1, 26 * 26 * 16])
		print ("Pool2_flat shape : ", pool2_flat.shape)

		dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
		print ("Dense shape : ", dense.shape)
	
		dropout = tf.layers.dropout (
				inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
		print ("Dropout shape : ", dropout.shape)

		logits = tf.layers.dense(inputs=dense, units=10)
		print ("Logits shape : ", logits.shape)
		
		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate Loss (for both TRAIN and EVAL modes)
		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
			train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Add evaluation metrics (for EVAL mode)
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=predictions["classes"])}
			
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



