from keras.layers import Input              # Input() is used to instantiate a Keras tensor.
from keras.layers import ZeroPadding2D      # ZeroPadding2D() is used to add zero-padding to the input.
from keras.layers import Conv2D             # Conv2D() is used to apply a convolution to the input.
from keras.layers import BatchNormalization # BatchNormalization() is used to normalize the input.
from keras.layers import Activation         # Activation() is used to apply an activation function to the input.
from keras.layers import MaxPooling2D       # MaxPooling2D() is used to apply a max-pooling operation to the input.
from keras.layers import AveragePooling2D   # AveragePooling2D() is used to apply an average-pooling operation to the input.
from keras.layers import Flatten            # Flatten() is used to flatten the input.
from keras.layers import Dense              # Dense() is used to add a fully connected layer to the network.

from keras.models import Model              

from keras.initializers import glorot_uniform # glorot_uniform is used to initialize weights.

from blocks import convolutional_block, identity_block

def ResNet50(input_shape, classes_count):
    X_input = Input(input_shape)
    X = ZeroPadding2D(padding=(3, 3))(X_input)

    # Stage 1 -> 1 layer
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2 -> 9 layers
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    # Stage 3 -> 12 layers
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4 -> 18 layers
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='f')

    # Stage 5 -> 9 layers
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, f = 3, filters = [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f = 3, filters = [512, 512, 2048], stage=5, block='c')

    # AVGPOOL -> 1 layer
    X = AveragePooling2D(pool_size=(2,2), name="avg_pool")(X)

    X = Flatten()(X)
    X = Dense(classes_count, activation='softmax', name='fc' + str(classes_count), kernel_initializer = glorot_uniform(seed=0))(X)

    # 1 + 9 + 12 + 18 + 9 + 1 = 50 layers
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

model = ResNet50(input_shape = (64, 64, 3), classes_count = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()