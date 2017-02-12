import numpy as np
import keras

from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers import Dense, Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Conv-ReLU-Pooling-Dense-ReLU-Dense-Softmax

def SimpleCNN():
    model = Sequential()
    model.add(Convolution2D(30, 5, 5,
                            border_mode = 'same',
                            input_shape = (1, 28, 28))
    )
    model.add(Activation('relu')
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activationo('softmax'))

    return model

    
