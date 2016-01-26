from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import time
import numpy as np
import sklearn.cross_validation
import sklearn.metrics
import theano
import theano.tensor as T
import lasagne
import sys, os.path
from scipy.misc import imresize, imread
from random import randint


RATIO = 0.6 # The ratio of the data set to use for training
PER_CATEGORY = 98 # Images to be used per category (training + validation)
CATEGORIES = 9 # Number of categories present in the data folder
DIR = "../wholedataset" # Path to folder
TYPE = ".jpg" # Extension of the images in the subfolders

DIM = 128 # Input to the network (images are resized to be square)
PREAUG_DIM = 140 # Dimensions to augment from

EPOCHS = 300
BATCH_SIZE = 1

SEED1 = 6789
SEED2 = 9876

SAVE = False

l2_regularization_rate = 0.0001

def build_model():
  """Returns the input and output layers in a tuple"""
# - conv layers take in 4-tensors with the following dimensions:
#   (batch size, number of channels, image dim 1, image dim 2)
  l_in = lasagne.layers.InputLayer(
      shape=(None, 3, DIM, DIM),
      )

  l_pad1 = lasagne.layers.PadLayer(
      l_in,
      width=1,#padding width
      )

  l_conv1 = lasagne.layers.Conv2DLayer(
      l_pad1,
      num_filters=32,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )

  l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, 
      pool_size=(2, 2),
      stride=2,
      )


  l_pad2 = lasagne.layers.PadLayer(
      l_pool1,
      width=1,#padding width
      )

  l_conv2 = lasagne.layers.Conv2DLayer(
      l_pad2,
      num_filters=64,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )

  l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, 
      pool_size=(2, 2),
      stride=2,
      )



  l_pad3 = lasagne.layers.PadLayer(
      l_pool2,
      width=1,#padding width
      )

  l_conv3 = lasagne.layers.Conv2DLayer(
      l_pad3,
      num_filters=128,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )


  l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, 
      pool_size=(2, 2),
      stride=2,
      )


  l_hidden1 = lasagne.layers.DenseLayer(
      l_pool3,
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      )

  l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)


  l_hidden2 = lasagne.layers.DenseLayer(
      l_hidden1_dropout,
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      )

  l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

# - applies the softmax after computing the final layer units
  l_out = lasagne.layers.DenseLayer(
      l_hidden2_dropout,
      #l_pool3,
      num_units=CATEGORIES,
      nonlinearity=lasagne.nonlinearities.softmax,
      #W=lasagne.init.GlorotUniform(),
      )

  return l_in, l_out
