# Author: Basharat Basharat
# Date  : 30-11-2018

import os
import shutil
import tensorflow as tf
import numpy as np
from time import time

import pickle
import gzip
from utils import *
import matplotlib.pyplot as plt
from tensorboard_evaluation import Evaluation


class Model (object):

	def __init__(self, history_length, learning_rate):


		self.learning_rate = learning_rate 
		
		# Placeholders for features and labels

		self.x  = tf.placeholder(tf.float32, shape = [None, 96, 96, history_length], name = "x")
		self.y_ = tf.placeholder(tf.float32, shape = [None,3], name = "y")

		#inputlayer = tf.reshape(features["x"], [-1, 96, 96, 1])

		conv1 = tf.layers.conv2d (inputs = self.x, filters = 32, kernel_size = 5, strides=2, padding = "same", activation = tf.nn.relu)
		conv2 = tf.layers.conv2d (inputs = conv1, filters = 32, kernel_size = 5, strides=2, padding = "same", activation = tf.nn.relu)
		conv3 = tf.layers.conv2d (inputs = conv2, filters = 16, kernel_size = 5, strides=1, padding = "same", activation = tf.nn.relu)
		print ("conv3 shape  : ", conv3.shape)

		conv3_flat = tf.reshape(conv3, [-1, 24*24*16])

		dense1 = tf.layers.dense(inputs=conv3_flat, units=256, activation=tf.nn.relu)
		##dropout1 = tf.layers.dropout(inputs=dense1, rate=0.7)
		##dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu)
		dropout2 = tf.layers.dropout(inputs=dense1, rate=0.7)


		self.logits = tf.layers.dense(inputs=dropout2, units=3, activation=None)
		print ("Logits shape : ", self.logits.shape)
		 
		# LSTM layer
		##lstm = tf.nn.rnn_cell.LSTMCell(num_units=128)
		##lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.7)
		##lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm])

		##init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
		##lstm_input = tf.expand_dims(self.logits, axis=1)

		##outputs, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=lstm_input, initial_state=init_state)
		##cell_out = tf.reshape(outputs, [-1, 128], name='flatten_lstm_outputs')
		##print ("cell_out shape : ", cell_out.shape)

		# output layer:
		#self.output = tf.contrib.layers.fully_connected(cell_out, 4, activation_fn=None)
		# TODO: Loss and optimizer
		#self.cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.output, labels=self.y_))
		#self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		#self.cost = tf.square (self.y_ - self.output) 
		#self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		print ("y_ shape : ", self.y_.shape)
		self.cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.logits, labels=self.y_))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

		self.y_pred = tf.nn.softmax(self.logits)
		#self.accuracy = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.y_,1))

		# TODO: Start tensorflow session		 
		self.sess = tf.Session()
		self.saver = tf.train.Saver()

	def training (self, X_train, y_train, X_valid, y_valid, epochs, batch_size, model_dir, tensorboard_dir):


		tensorboard_eval = Evaluation(tensorboard_dir)
	
		acc = []
		acc_valid = []

		train_cost = np.zeros(epochs)
		valid_cost = np.zeros(epochs)

		correct_prediction = self.y_pred 
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		#accuracy = self.accuracy

		#tf.reset_default_graph()
		with self.sess as sess:
			sess.run(tf.global_variables_initializer())
			print('inside session')
			start_time = time()
    
			for epoch in range(epochs):
				correctsum = 0
				#x_train_batchsize = 16
				x_train_batchsize = len(X_train)//batch_size
				print("X_train_batchsize :::::", x_train_batchsize)

				correctsum_val = 0
				#x_val_batchsize = 12
				x_val_batchsize = len(X_valid)//batch_size 
				    
				train_cost = np.zeros((epochs))
				valid_cost = np.zeros((epochs))


				# for Training computation
				for i  in range(x_train_batchsize):

					x_batch = X_train[i*batch_size:(i+1)*batch_size]
					y_batch = y_train[i*batch_size:(i+1)*batch_size]

					y_batch = id_to_action(y_batch)
					_, loss = sess.run([self.optimizer, self.cost],feed_dict={self.x:x_batch, self.y_: y_batch})
					train_cost[epoch] += sess.run(self.cost, feed_dict={self.x: x_batch, self.y_: y_batch})
					print('iteration {} , epoch {}, train cost {:.2f}'.format(i,epoch,train_cost[epoch]))

				print('epoch {}, loss {:.2f} %'.format(epoch, loss), end='\n')


				# for Validation computation
				for i  in range(x_val_batchsize):
					x_val = X_valid[i*batch_size:(i+1)*batch_size]
					y_val = y_valid[i*batch_size:(i+1)*batch_size]
					#y_val = id_to_action(y_val)
					
					valid_cost[epoch] += self.sess.run(self.cost, feed_dict={self.x:x_val, self.y_:y_val})
					print('valid iteration{}, epoch{}, valid cost ::: {:.2f}'.format(i,epoch,valid_cost[epoch]))

				train_cost[epoch] = train_cost[epoch] / x_train_batchsize
				valid_cost[epoch] = valid_cost[epoch] / x_val_batchsize
				print("[%d/%d]: train_cost: %.4f, valid_cost: %.4f" %(epoch+1, epochs, train_cost[epoch], valid_cost[epoch]))
				print('epoch {},train_cost{:.2f}, validation cost{:.2f}'.format(epoch,train_cost[epoch], valid_cost[epoch]))
				eval_dict = {"train":train_cost[epoch], "valid":valid_cost[epoch]}
				tensorboard_eval.write_episode_data(epoch, eval_dict)


			# TODO: save your self
			self.save(os.path.join(model_dir, "self.ckpt"))
			print(model_dir)
			print("Model saved in file: %s" % model_dir)
			self.sess.close()
			return eval_dict

	def load(self, file_name):
		self.saver.restore(self.sess, file_name)

	def save(self, file_name):
		self.saver.save(self.sess, file_name)

