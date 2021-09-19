import utils

def convolutional_block(X):
    X_skip = X

    X = Conv2D()(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D()(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D()(X)
    X = BatchNormalization()(X)

    X_skip = Conv2D()(X_skip)
    X_skip = BatchNormalization()(X_skip)

    X = Add()([X, X_skip])
    X = Activation('relu')(X)
    
    return X

def identity_block(X):
    X_skip = X

    X = Conv2D()(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D()(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D()(X)
    X = BatchNormalization()(X)

    X = Add()([X, X_skip])
    X = Activation('relu')(X)
    
    return X