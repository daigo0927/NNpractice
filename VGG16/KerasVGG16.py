import numpy as np

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def KerasVGG16(input_shape = (3, 224, 224)):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3,
                            border_mode = 'same',
                            input_shape = input_shape,
                            activation = 'relu'))
    model.add(Convolution2D(64, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Convolution2D(128, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    model.add(Convolution2D(128, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Convolution2D(256, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    model.add(Convolution2D(256, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    model.add(Convolution2D(256, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Convolution2D(512, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    model.add(Convolution2D(512, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    model.add(Convolution2D(512, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Convolution2D(512, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    model.add(Convolution2D(512, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    model.add(Convolution2D(512, 3, 3,
                            border_mode = 'same',
                            activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation = 'softmax'))

    return model
    
