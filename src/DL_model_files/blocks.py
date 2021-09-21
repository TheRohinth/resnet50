from keras.layers import Conv2D             # Conv2D() is used to apply a convolution to the input.
from keras.layers import BatchNormalization # BatchNormalization() is used to normalize the input.
from keras.layers import Activation         # Activation() is used to apply an activation function to the input.
from keras.layers import Add                # Add() is used to add two tensors.

from keras.initializers import glorot_uniform # glorot_uniform is used to initialize weights.

def convolutional_block(X, f, filters, stage, block, s = 2):

    '''
        Implementation of the convolutional block

        Arguments:
            X -> input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -> integer, specifying the shape of the middle CONV's window for the main path
            filters -> python list of integers, defining the number of filters in the CONV layers of the main path
            stage -> integer, used to name the layers, depending on their position in the network
            block -> string/character, used to name the layers, depending on their position in the network
            s -> Integer, specifying the stride to be used
    
        Returns:
            X -> output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    '''

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_skip = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # SHORTCUT PATH
    X_skip = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_skip)
    X_skip = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_skip)

    # Final step
    X = Add()([X, X_skip])
    X = Activation('relu')(X)
    
    return X

def identity_block(X, f, filters, stage, block):

    '''
        Implementation of the identity block

        Arguments:
            X -> input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -> integer, specifying the shape of the middle CONV's window for the main path
            filters -> python list of integers, defining the number of filters in the CONV layers of the main path
            stage -> integer, used to name the layers, depending on their position in the network
            block -> string/character, used to name the layers, depending on their position in the network  
        
        Returns:
            X -> output of the identity block, tensor of shape (n_H, n_W, n_C)
    '''

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_skip = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step
    X = Add()([X, X_skip])
    X = Activation('relu')(X)
    
    return X