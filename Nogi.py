# coding utf-8
import numpy as np
import gzip
from PIL import Image, ImageOps
import pickle
import pdb
from keras.utils import np_utils

# in use, put NogiFaceData.pkl, NogiFaceLabel.pkl in a directory : dir
# need keras installation
# > from Nogi import NogiFaceImport
# > nogi = NogiFaceImport(path = 'path/to/dir/')
# with load, put appropriate args
# > (x_train, t_train), (x_test, t_test) = nogi.load()

class NogiFaceImport:
    
    def __init__(self,
                 import_type = '.pkl',
                 path = './'):
        
        path = path

        print('loading Nogizaka46 Face Data ...')
        if import_type == '.pkl':
            with open(path + 'NogiFaceData' + import_type, 'rb') as f:
                self.data_origin = pickle.load(f)
            with open(path + 'NogiFaceLabel' + import_type, 'rb') as f:
                self.label_origin = pickle.load(f)
            
        elif import_type == '.gz':
            with gzip.open(path + 'NogiFaceData' + import_type, 'rb') as f:
                self.data_origin = f.read()
            with gzip.open(path + 'NogiFaceLabel' + import_type, 'rb') as f:
                self.label_origin = f.read()

        else:
            print('con\'t load any data, set .pkl or .gz')

    def load(self,
             front_only = False,        # except others
             image_size = (128, 128),   # image size
             gray = False,              # grayscale
             one_hot = True,            # make labels one-hot vector
             train_ratio = 0.9):        # train/test split ratio
        
        data = self.data_origin
        label = self.label_origin
        
        img_num, origin_H, origin_W, color_num = data.shape

        # originally shape(256, 256, 3)
        # if needed, resize
        resizer = image_size
        if resizer != (origin_H, origin_W):
            data_resized = np.array([np.array(Image.fromarray(arr).resize(resizer))
                                     for arr in data])
            data = data_resized
            
        # data contains 5 each labeled member, and others
        # Asuka Saito, Erika Ikuta, Mai Shiraishi, Nanase Nishino, Nanami Hashimoto, and others
        # if needed, except others
        if front_only == True:
            front_idx = label!=5
            
            data = data[front_idx]
            label = label[front_idx]

        # if needed, convert grayscale
        if gray == True:
            data_grayed = np.array([np.array(ImageOps.grayscale(Image.fromarray(arr)))
                                    for arr in data])
            data = data_grayed

        # if needed, convert label to one-hot vector
        if one_hot == True:
            label_num = max(label)+1

            label = np_utils.to_categorical(label, label_num)

        # shuffle data
        img_num = data.shape[0]
        shuffle_idx = np.random.permutation(img_num)

        # split train and test data
        train_idx = shuffle_idx[:np.int(img_num * train_ratio)]
        data_train = data[train_idx]
        label_train = label[train_idx]
        
        test_idx = shuffle_idx[np.int(img_num * train_ratio):]
        data_test = data[test_idx]
        label_test = label[test_idx]

        print('front only : ' + str(front_only))
        print('H/W size : ' + str(resizer))
        print('grayscale : ' + str(gray))
        print('one hot : ' + str(one_hot))
        print('train data shape : ' + str(data_train.shape) \
              + ', train label shape : ' + str(label_train.shape))
        print('test data shape : ' + str(data_test.shape) \
              + ', test label shape : ' + str(label_test.shape))
        
        return (data_train, label_train), (data_test, label_test)
