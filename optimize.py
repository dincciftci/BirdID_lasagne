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
import argparse
import multiprocessing
import simple_spearmint

# ############################### prepare data ###############################

parser = argparse.ArgumentParser(description= 'Accept network configuration and save file')
parser.add_argument("-c", "--config", default= "defaultconfig", help="The name of the network configuration to be used")
parser.add_argument("-s", "--save", default= "", help= "The name of the file that the trained network will be saved in")
parser.add_argument("-r", "--resume", default= "", help= "The .npy file containing the values of the parameters for the network being resumed (include the extension)")
args = parser.parse_args()

print("Using the configuration from " + args.config + ".py")

config = __import__(args.config)

def addFiles(file, DIR, foldername, num, PREAUG_DIM):
  img = imread(DIR +"/" + foldername + "/" + file)
  img = imresize(img, (PREAUG_DIM, PREAUG_DIM))
  return [num, img]

worker = __import__("train_net_args")

# will take to the power of 10 to use
parameter_space = {'lr': {'type': 'float', 'min': -6, 'max': 1},
    'l2': {'type': 'int', 'min': -6, 'max': 1}}

# Create an optimizer
ss = simple_spearmint.SimpleSpearmint(parameter_space)

