"""
    2019 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:

    Y.Niu, B. Tondi, Y.Zhao, M.Barni:
    “Primary Quantization Matrix Estimation of Double Compressed JPEG Images via CNN",
    IEEE Signal Processing Letters, 2019, November
    Available on ArXiv: arXiv preprint:1908.04259  https://arxiv.org/abs/1908.04259

"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K


def custom_qf_loss(y_true, y_pred):

    """ Keras custom loss for QF Estimation

    :param y_true: Tensorflow/Theano tensor of predicted labels
    :param y_pred: Tensorflow/Theano tensor of true labels
    :return: Custom loss
    """
    return K.mean(K.pow(2*K.abs(y_pred - y_true), 20) / (1 + K.pow(2*K.abs(y_pred - y_true), 19)))


def normalised_mean_squared_error(y_true, y_pred):
    """ Keras custom loss: Normalised Mean Squared Error

    :param y_true: Tensorflow/Theano tensor of predicted labels
    :param y_pred: Tensorflow/Theano tensor of true labels
    :return: Normalised Mean Squared Erro
    """
    return K.mean(K.square(y_pred - y_true) / K.square(y_true), -1)


def change_last_layer_nclasses(model, num_classes, freeze=False):

    """ Changes last (Dense) layer of ContrastNet for transfer learning

    Args:
       model: ContrastNet model.
       num_classes: number of new output classes

    Returns:
       New Keras sequential model.
    """

    for layer in model.layers:
        layer.trainable = not freeze

    # define a new output layer to connect with the last fc layer in vgg
    x = model.layers[-2].output
    new_output_layer = Dense(num_classes, activation='relu', name='predictions')(x)

    # combine the original VGG model with the new output layer
    new_model = Model(inputs=model.input, outputs=new_output_layer)

    return new_model


def contrastNet(in_shape=(64, 64, 3), num_classes=2, nf_base=64, layers_depth=(4, 3)):

    """ Builds the graph for a CNN based on Keras (TensorFlow backend)

    Args:
       in_shape: the shape on the input image (Height x Width x Depth).
       num_classes: number of output classes
       nf_base: number of filters in the first layer
       layers_depth: number of convolutions at each layer

    Returns:
       Keras sequential model.
    """

    model = Sequential()

    # First convolution and Max Pooling (ReLu activation)
    model.add(Conv2D(nf_base, kernel_size=(3, 3), strides=(1, 1), input_shape=in_shape, activation='relu', name='conv1_1'))

    for i in range(0, layers_depth[0]):
        model.add(Conv2D(nf_base+nf_base*(i+1),
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         activation='relu',
                         name='conv1_{}'.format(i+2)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    last_size = nf_base+nf_base*(i+1)

    # Second convolution and Max Pooling (ReLu activation)
    for i in range(0, layers_depth[1]):
        model.add(Conv2D(last_size+nf_base*(i+1),
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         activation='relu',
                         name='conv2_{}'.format(i+2)))

    # Third convolution and Max Pooling (ReLu activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    nf = int(model.layers[-1].output_shape[-1]/2)

    model.add(Conv2D(nf, kernel_size=(1, 1), strides=1, name='conv3_1'))

    # Flatten before fully-connected layer(s)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='relu', name='predictions'))

    return model
