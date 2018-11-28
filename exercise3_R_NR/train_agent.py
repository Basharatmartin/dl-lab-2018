from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from kmodel import Model
import tensorflow as tf
from utils import *
from tensorboard_evaluation import Evaluation


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    X_train = rgb2gray(X_train)
    X_train = np.expand_dims(X_train, axis=3)
    X_valid = rgb2gray(X_valid)
    X_valid = np.expand_dims(X_valid, axis=3)
    ##y_valid = rgb2gray(y_valid)



    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(32, 16, history_length=1)
    training_loss = agent.train_valid(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr)
    
    tensorboard_eval = Evaluation(tensorboard_dir)


    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     tensorboard_eval.write_episode_data(...)
      
    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    # print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    #print (X_train)
    #print (X_train.shape)
    # train model (you can change the parameters!)
    
    #print (X_train[2625])

    ##for i in range(2615,2640):
    ##    print (y_train[i])
    ##    plt.imshow(X_train[i], cmap='gray')
    ##    plt.show()

    
    ##train_model(X_train, y_train, X_valid, n_minibatches=100000, batch_size=64, lr=0.0001)
    train_model(X_train, y_train, X_valid, y_valid, 5, batch_size=64, lr=0.01)
    print ("y_train shape ::: ", y_train.shape)
