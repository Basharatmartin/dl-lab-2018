### Author : Basharat Basharat
### Matr.Nr: 4053100
### Email  : basharat.basharat@uranus.uni-freiburg.de

#### Tensorflow class for mnist images data x=(50000, 28x28, 1) y=(0~9)

from __future__ import print_function

import tensorflow as tf
import tensorflow.layers as layers
import numpy as np

class CnnFlow ():

	def __init__ (self, nr_filters, filter_size):
		self.nr_filters = nr_filters
		self.filter_size = filter_size

		## after reading over the tensorflow website tutorial
		## placeholder for the data (x)
		self.inputlayer = tf.placeholder (tf.float32, shape=[None, 28, 28, 1])
		
		## convolution layer 1
		conv1 = layers.Conv2D (	filters=nr_filters,
					kernel_size = filter_size,
					padding = "same",
					activation ="relu")
		
		pool1 = layers.MaxPooling2D(pool_size=2, strides=1)

		conv2 = layers.Conv2D (	filters = nr_filters,
					kernel_size = filter_size,
					padding = "same",
					activation = "relu")

		pool2 = layers.MaxPooling2D (pool_size=2, strides=1)
		flatten = layers.Flatten()	
		dense = layers.Dense(units=128, activation="relu")
		self.logits = layers.Dense(units=10)

		self.x_layer = conv1(self.inputlayer)
		for x in [pool1, conv2, pool2, dense, self.logits]:
			self.x_layer = x(self.x_layer)

		y_pred = tf.nn.softmax(self.logits)

		### Tensorflow variables initialization
		start = tf.global_variabls_initializer()
		self.session = tf.Session()
		self.session.run(start)

	def training (self, x_train, y_train, x_valid, y_valid, num_epochs, lr, batch_size=200):

		## placeholder for Y_true and in onehot data
		Y_true = tf.placeholder (tf.float32, shape=[None, 10])
		loss = tf.losses.softmax_cross_entropy(onehot_labels=Y_true, logits=self.logits)
		train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
		training_loss = 0
		validation_error = np.empty((num_epochs))
		validation_error[:] = np.nan

		### mini batch mechanisim from the exercise nr. 1

		nr_samples = x_train.shape[0]
		nr_batches = nr_samples // batch_size

		print ("...Starting training")
		
		for e in range (num_epochs):
			for b in range(nr_batches):
				### for training data loss
				x_batch = x_train[b*batch_size : (b+1)*batch_size]
				y_batch = y_train[b*batch_size : (b+1)*batch_size]
				

				_, loss_value = self.session.run([train_op, loss], {self.inputlayer: x_batch, Y_true: y_batch})
				training_loss = training_loss + loss_value
				training_loss = np.average(training_loss)

			#### for validation loss
			validation_loss = self.session.run(loss, {self.inputlayer: x_valid, Y_true : y_valid})
		
			training_accuracy = self.accuracy (x_train, y_train)
			validation_accuracy = self.accuracy (x_valid, y_valid)

			print('epoch {:.4f}, train loss {:.4}, train acc {:.4f}'.format(e, training_loss, training_accuracy))
			print('epoch {:.4f}, valid loss {:.4}, valid acc {:.4f}'.format(e, validation_loss, validation_accuracy))
			validation_error(1-validation_accuracy)
		return validation_error

	def predictition(self, x_train):
		Y_pred = self.session.run(self.y_pred(x_train))
		return Y_pred

	def accuracy(self, X_data, Y_true):
		Y_pred = self.session.run(self.xlayer, {self.inputlayer: x})
		Y_pred_index = tf.argmax(Y_pred, axis=1)
		Y_true_index = tf.argmax(Y_true, axis=1)
		correct = np.sum (Y_pred_index == Y_true_index)

		return (accuray / x_train.shape[0])

	
