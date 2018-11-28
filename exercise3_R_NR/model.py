from __future__ import print_function

import tensorflow as tf
import tensorflow.layers as layers

class Model(object):

	def __init__(self, nr_filters, kernel_size):

		self.inputs = tf.placeholder (tf.float32, shape=[None, 96, 96, 1])

		print ("self.inputs shape :::: ", self.inputs.shape)
		## first conv layer
		conv1 = layers.Conv2D(	filters=nr_filters,
					kernel_size=kernel_size,
					padding="same",
					activation="relu")
		## pooling 1
		pooling = layers.MaxPooling2D(pool_size=2, strides=2)
		
		## conv2
		conv2 = layers.Conv2D(	filters=nr_filters,
					kernel_size=kernel_size,
					padding="same",
					activation="relu")
		
		## flatten the layer
		flatten = layers.Flatten()
		
		linear1 = layers.Dense(units=64, activation="relu")
		linear2 = layers.Dense(units=3, activation=None)
		
		self.logits = conv1(self.inputs)

		for layer in [pooling, conv2, pooling, flatten, linear1, linear2] :
			self.logits = layer(self.logits)
		
		self.out_soft = tf.nn.softmax(self.logits)

		print ("self.out_soft shape :::: ", self.out_soft)
		

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

		# ...

		# TODO: Start tensorflow session
		# ...

		#self.saver = tf.train.Saver()


	def load(self, file_name):
		self.saver.restore(self.sess, file_name)

	def save(self, file_name):
		self.saver.save(self.sess, file_name)

	def train (self, x_train, y_train, epochs, batch_size, lr):
		
		self.y_true = tf.placeholder (tf.float32, shape=[None, 3])
		print ("self.y_true shape :::: ", self.y_true.shape)

		self.loss = tf.losses.softmax_cross_entropy(self.y_true, self.logits)
		


		optimizer = tf.train.GradientDescentOptimizer (learning_rate=lr)
		train = optimizer.minimize(self.loss)

		## mini batches
		## leave for now !!!!



		num_batches = x_train.shape[0] // batch_size
		val_error = []


		for i in range(epochs):
			train_loss = 0
			for b in range (num_batches):

				x_batch = x_train[b*batch_size:b*batch_size+batch_size]
				y_batch = y_train[b*batch_size:b*batch_size+batch_size]

				_, loss_value = self.sess.run ([train, self.loss], {self.inputs:x_batch, self.y_true:y_batch})
				train_loss +=loss_value


			val_loss = self.sess.run(self.loss, {self.inputs: x_valid, self.y_true: y_valid})
		 	
			##val_acc = self.accuracy(x_valid, y_valid)
		 	##train_acc = self.accuracy(x_train, y_train


			print ('Epoch : %i Train_loss : %.4f' (i, train_loss))

			print('Epoch: %i Validation loss: %.4f' % (i, val_loss))

		return train_loss




			



