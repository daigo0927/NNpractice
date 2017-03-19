from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST dataset with easy shape
def loadMNIST():

    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

    nb_classes = 10
    t_train = np_utils.to_categorical(t_train, nb_classes)
    t_test = np_utils.to_categorical(t_test, nb_classes)

    print('data shape : (N, 1, 28, 28)')
    return((x_train, t_train), (x_test, t_test))
