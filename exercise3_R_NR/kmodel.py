from __future__ import print_function

import tensorflow as tf
import numpy as np

def reset_graph(seed=42):
	tf.reset_default_graph()
	tf.set_random_seed(seed)
	np.random.seed(seed)


class Model(object):

	def __init__(self, nr_filters, kernel_size, history_length=1):
		
		reset_graph()

		self.inputlayer = tf.placeholder (tf.float32, shape=[None, 96, 96, history_length])
		conv1 = tf.layers.conv2d (
					inputs = self.inputlayer,
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

		pool2_flat = tf.reshape(pool2, [-1, 94 * 94 * 16])
		print ("Pool2_flat shape : ", pool2_flat.shape)

		dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
		print ("Dense shape : ", dense.shape)
	
		self.logits = tf.layers.dense(inputs=dense, units=3)
		print ("Logits shape : ", self.logits.shape)

		self.softout = tf.nn.softmax(self.logits)	
		print ("Softout shape : ", self.softout.shape)

		self.y_true = tf.placeholder (tf.float32, shape=[None, 3])
		print ("self.y_true shape :::: ", self.y_true.shape)
		self.loss = tf.losses.softmax_cross_entropy(self.y_true, self.logits)
		self.train = optimizer.minimize(self.loss)
		
		
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def train_valid (self, x_train, y_train, x_valid, y_valid, epochs, batch_size, lr):

		
		
		
		##optimizer = tf.train.GradientDescentOptimizer (learning_rate=lr)

		num_batches = x_train.shape[0] // batch_size
		val_error = []
		
		for i in range(epochs):
			train_loss = 0
			for b in range (num_batches):

				x_batch = x_train[b*batch_size:b*batch_size+batch_size]
				y_batch = y_train[b*batch_size:b*batch_size+batch_size]

				_, loss_value = self.sess.run ([self.train, self.loss], {self.inputlayer:x_batch, self.y_true:y_batch})
				train_loss +=loss_value


			val_loss = self.sess.run(self.loss, {self.inputlayer: x_valid, self.y_true: y_valid})
		 	
			##val_acc = self.accuracy(x_valid, y_valid)
		 	##train_acc = self.accuracy(x_train, y_train


			print ('Epoch : %i Train_loss : %.4f' %(i, train_loss))

			print('Epoch: %i Validation loss: %.4f' %(i, val_loss))

		return train_loss


	def load(self, file_name):
		self.saver.restore(self.sess, file_name)

	def save(self, file_name):
		self.saver.save(self.sess, file_name)



			



