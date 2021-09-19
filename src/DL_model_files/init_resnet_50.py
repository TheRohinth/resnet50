import utils
from blocks import convolutional_block, identity_block

def ResNet50(input_shape, classes):
    X_input = Input(input_shape)
    X = ZeroPadding2D()

    # 1 layer
    X = Conv2D()(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D()(X)

    # 9 layers
    X = convolutional_block()
    X = identity_block()
    X = identity_block()

    # 12 layers
    X = convolutional_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()

    # 18 layers
    X = convolutional_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()
    X = identity_block()

    # 9 layers
    X = convolutional_block()
    X = identity_block()
    X = identity_block()

    # 1 layer
    X = AveragePooling2D()(X)

    X = Flatten()(X)
    X = Dense(activation='softmax')(X)

    # 1 + 9 + 12 + 18 + 9 + 1 = 50
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model