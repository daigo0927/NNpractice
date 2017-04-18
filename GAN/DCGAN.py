# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.core import Permute
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D

def GeneratorModel():

    inputs = Input(shape=(100, ))
    x = Dense(1024)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128*8*8*3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((128*3, 8, 8))(x) 
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64*3, (5,5), padding = 'same')(x) # shape(64, 16, 16, 3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32*3, (5,5), padding = 'same')(x) # shape(32, 32, 32, 3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(1*3, (5,5), padding = 'same')(x) # shape(1, 64, 64, 3)
    x = Permute((1, 2, 0))(x)
    images = Activation('tanh')(x)

    model = Model(input = inputs, output = images)

    return model

def DiscriminatorModel():

    images = Input(shape = (64, 64, 3))
    x = Conv2D(64, (5, 5), strides = (2,2), padding = 'same')(images)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (5, 5), strides = (2,2), padding = 'same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(input = images, output = outputs)

    return model


    

    
    
    
    
    
    


