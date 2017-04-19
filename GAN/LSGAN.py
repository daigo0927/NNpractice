# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from PIL import Image
import h5py
import math

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.core import Permute
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam

def GeneratorModel():

    inputs = Input(shape=(100, ))
    x = Dense(1024)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128*8*8*3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((8, 8, 128*3))(x) 
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64*3, (5,5), padding = 'same')(x) # shape(None, 16, 16, 64*3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32*3, (5,5), padding = 'same')(x) # shape(None, 32, 32, 32*3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(1*3, (5,5), padding = 'same')(x) # shape(None, 64, 64, 1*3)
    images = Activation('tanh')(x) # shape(None, )

    model = Model(inputs = inputs, outputs = images)

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

    model = Model(inputs = images, outputs = outputs)

    return model


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((height*rows, width*cols, 3),
                              dtype=generated_images.dtype)
    
    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1), :] \
            = image
        
    return combined_image



BatchSize = 40
NumEpoch = 100

ResultPath = {}
ResultPath['image'] = './image/'
ResultPath['model'] = './model/'
for path in ResultPath:
    if not os.path.exists(path):
        os.mkdir(path)
        

def train(x_train):
    # x_train.shape(instance, 64, 64, 3)
    # not need test data, labels
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    d_model = DiscriminatorModel()
    d_opt = Adam(lr = 1e-5, beta_1 = 0.1)
    d_model.compile(loss = 'mean_squared_error',
                    optimizer = d_opt)

    g_model = GeneratorModel()
    lsgan = Sequential([g_model, d_model])
    g_opt = Adam(lr = 1e-5, beta_1 = 0.5)
    lsgan.compile(loss = 'mean_squared_error', optimizer = g_opt)
    # dcgan.summary()

    num_batches = int(x_train.shape[0]/BatchSize)
    print('Number of batches : {}'.format(num_batches))

    # '''
    for epoch in range(NumEpoch):

        for index in range(num_batches):

            noise = np.array([np.random.uniform(-1, 1, 100)\
                              for _ in range(BatchSize)])
            image_batch = x_train[index*BatchSize:(index+1)*BatchSize]

            generated_images = g_model.predict(noise, verbose = 0)

            x = np.concatenate((image_batch, generated_images))
            y = [1]*BatchSize + [0]*BatchSize
            d_loss = d_model.train_on_batch(x, y)

            # train generator once in 3 iteration
     
            noise = np.array([np.random.uniform(-1,1, 100) \
                              for _ in range(BatchSize)])
            g_loss = dcgan.train_on_batch(noise, [1]*BatchSize)

            if index == num_batches-1:
                image = combine_images(generated_images)
                image = image*127.5 + 127.5
                
                Image.fromarray(image.astype(np.uint8))\
                     .save(ResultPath['image'] + '{}.png'.format(epoch))

            print('epoch:{}, batch:{}, g_loss:{}, d_loss:{}'.format(epoch,
                                                                    index,
                                                                    g_loss,
                                                                    d_loss))

    g_model.save_weights(ResultPath['model'] + 'lsgan_g.h5')
    d_model.save_weights(ResultPath['model'] + 'lsgan_d.h5')
    # '''

    
    
    
    

    
    
    
    
    
    


