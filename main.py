







from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *



np.set_printoptions(threshold=np.nan)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())


def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.squared_difference(anchor, positive), axis = -1)
    neg_dist = tf.reduce_sum(tf.squared_difference(anchor, negative), axis = -1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss
	
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)


database = {}
database["siddharth"] = img_to_encoding("train/sid1.png", FRmodel)
database["jay"] = img_to_encoding("train/jay1.jpg", FRmodel)


def verify(image_path, identity, database, model):
    
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
        
        
    return dist, door_open
	
	
	
verify("images/sid1.jpg", "Siddharth", database, FRmodel)
verify("images/jay.jpg", "Jay", database, FRmodel)

def who_is_it(image_path, database, model):
	encoding = img_to_encoding(image_path, model)
	min_dist = 100
	for (name) in database:
	    dist = np.linalg.norm(encoding - database[name])
	    if dist < min_dist:
	        min_dist = dist
	        identity = name

	if min_dist > 0.7:
	    print("Not in the database.")
	else:
	    print ("it's " + str(identity) + ", the distance is " + str(min_dist))
	    
	return min_dist, identity
	
who_is_it("images/camera_0.jpg", database, FRmodel)
