import utils
from blocks import convolutional_block, identity_block

def ResNet50(input_shape, classes):
    X_input = Input(input_shape)
    X = ZeroPadding2D()

    X = Conv2D()(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D()(X)

    X = convolutional_block()
    X = identity_block()
    X = identity_block()

    X = convolutional_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()

    X = convolutional_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()

    X = convolutional_block()
    X = identity_block()
    X = identity_block()

    X = AveragePooling2D()(X)

    X = Flatten()(X)
    X = Dense(activation='softmax')(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model