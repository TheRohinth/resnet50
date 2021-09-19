# keras.layers --------------------------------------

from keras.layers import Input              # Input() is used to instantiate a Keras tensor.
from keras.layers import ZeroPadding2D      # ZeroPadding2D() is used to add zero-padding to the input.
from keras.layers import Conv2D             # Conv2D() is used to apply a convolution to the input.
from keras.layers import BatchNormalization # BatchNormalization() is used to normalize the input.
from keras.layers import Activation         # Activation() is used to apply an activation function to the input.
from keras.layers import MaxPooling2D       # MaxPooling2D() is used to apply a max-pooling operation to the input.
from keras.layers import AveragePooling2D   # AveragePooling2D() is used to apply an average-pooling operation to the input.
from keras.layers import Flatten            # Flatten() is used to flatten the input.
from keras.layers import Dense              # Dense() is used to add a fully connected layer to the network.

# keras.models --------------------------------------

from keras.models import Model              # Model() is used to create a model.