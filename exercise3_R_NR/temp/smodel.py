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


class Model (object):

	def __init__(self, history_length, learning_rate):


		self.learning_rate = learning_rate 
		
		# Placeholders for features and labels

		self.x  = tf.placeholder(tf.float32, shape = [None, 96, 96, history_length], name = "x")
		self.y_ = tf.placeholder(tf.float32, shape = [None,3], name = "y")

		#inputlayer = tf.reshape(features["x"], [-1, 96, 96, 1])

		conv1 = tf.layers.conv2d (inputs = self.x, filters = 16, kernel_size = 5, strides=2, padding = "same", activation = tf.nn.relu)
		conv2 = tf.layers.conv2d (inputs = conv1, filters = 32, kernel_size = 5, strides=2, padding = "same", activation = tf.nn.relu)
		conv3 = tf.layers.conv2d (inputs = conv2, filters = 32, kernel_size = 5, strides=1, padding = "same", activation = tf.nn.relu)

		conv3_flat = tf.reshape(conv3, [-1, 96*96*32])

		dense1 = tf.layers.dense(inputs=conv3_flat, units=256, activation=tf.nn.relu)
		dropout1 = tf.layers.dropout(inputs=dense1, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)
		dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)
		dropout2 = tf.layers.dropout(inputs=dense2, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)


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
		self.cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.logits, labels=self.y_))
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

		self.y_pred = tf.nn.softmax(self.logits)
		self.accuracy = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.y_,1))

		# TODO: Start tensorflow session		 
		self.sess = tf.Session()
		self.saver = tf.train.Saver()

	def load(self, file_name):
		self.saver.restore(self.sess, file_name)

	def save(self, file_name):
		self.saver.save(self.sess, file_name)

