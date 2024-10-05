from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import UpSampling2D, concatenate, add
from keras.layers import Concatenate, BatchNormalization
from keras import backend as K


def conv_block(inp, encoder_list, channels, batchnorm, regularization, encoder_branch=False, pool=False):
    """
    Convolutional block, consists of optional max-pooling followed by two convolutional layers

    :param inp: Layer before convolutional block
    :param encoder_list: List where layer outputs for cross concatenation are saved
    :param channels: Number of channels within the block
    :param batchnorm: Optional batch-normalization
    :param regularization: Optional kernel regularization
    :param encoder_branch: Boolean, is block in the encoder branch or not
    :param pool: Boolean, whether to apply max-pooling before convolutional block
    :return: x: last (batch-normalized) convolutional layer and encoder_list: see encoder_list
    """
    # Apply max-pooling before convolutions, if specified (only in down-branch and bottom block)
    if pool:
        inp = MaxPooling2D()(inp)

    # Convolutional layer followed by optional batch-normalization
    x = Conv2D(channels, 3, activation='relu', padding='same', kernel_regularizer=regularization)(inp)
    x = BatchNormalization()(x) if batchnorm else x

    x = Conv2D(channels, 3, activation='relu', padding='same', kernel_regularizer=regularization)(x)
    x = BatchNormalization()(x) if batchnorm else x

    # Save the last Conv2D layer output for cross connections
    if encoder_branch:
        encoder_list.append(x)

    return x, encoder_list


def encoder_path(inp, channels, depth, batchnorm, regularization):
    """
    Encoder path of the U-Net, creates the down-branch

    :param inp: Input layer
    :param channels: Number of initial channels
    :param depth: Total depth of the U-Net
    :param batchnorm: Optional batch-normalization
    :param regularization: Optional kernel regularization
    :return: encoder: last layer of the down-branch and encoder_list: output of each block for cross connections
    """
    encoder_list = []
    encoder = inp
    for l_idx in range(depth):
        if l_idx == 0:
            encoder, encoder_list = conv_block(encoder, encoder_list, channels, batchnorm,
                                                 regularization, encoder_branch=True)
        else:
            encoder, encoder_list = conv_block(encoder, encoder_list, channels, batchnorm,
                                                 regularization, encoder_branch=True, pool=True)

        # Increase number of channels in deeper layers
        channels *= 2

    return encoder, encoder_list


def decoder_path(inp, encoder_list, channels, depth, batchnorm, regularization):
    """
    Decoder path of the U-Net, creates the up-branch

    :param inp: Last layer from the bottom of the U-Net
    :param encoder_list: outputs from the encoder path to form the cross connections
    :param channels: Maximum number of channels (equal to initial_channels * 2^depth)
    :param depth: Total depth of the U-Net
    :param batchnorm: Optional batch-normalization
    :param regularization: Optional kernel regularization
    :return: Last layer of the up-branch
    """
    decoder = inp
    for l_idx in range(depth):
        # Decrease number of channels in higher layers
        channels = int(channels / 2)

        # Start of with upsampling, followed by convolution, followed by batch normalization
        decoder = Conv2D(channels, 2, activation='relu', padding='same', kernel_regularizer=regularization)(
            UpSampling2D(size=(2, 2))(decoder))
        decoder = BatchNormalization()(decoder) if batchnorm else decoder

        # Concatenate
        decoder = concatenate([encoder_list[-(l_idx + 1)], decoder], axis=3)

        # Convolutional block
        decoder, _ = conv_block(decoder, encoder_list, channels, batchnorm, regularization)

    return decoder


def unet(input_shape=(512, 512, 3), depth=4, channels=64, regularization=None, batchnorm=False):
    """
    This function can create a U-Net for any amount of inputs with any desired network depth.
    Additional functionalities like batch normalization, residual layers and dropout can be selected.

    :param input_shape: Shape of the input image
    :param depth: Depth of the network, number of down- and upsample blocks
    :param channels: Number of output filters in the first convolutional block
    :param output_channels: Number of channels of the output image
    :param regularization: Optional kernel regularizer
    :param batchnorm: Optional batch normalization
    :return: Keras model
    """

    # Input layer
    inp = Input(shape=input_shape)

    # Encoder path
    encoder, encoder_list = encoder_path(inp, channels, depth, batchnorm, regularization)

    # Bottom part, similar to encoder however output is not saved for U-Net concatenation
    channels = channels * 2**depth
    bottom, _ = conv_block(encoder, encoder_list, channels, batchnorm, regularization, pool=True)

    # Decoder path
    decoder = decoder_path(bottom, encoder_list, channels, depth, batchnorm, regularization)

    # Output layer
    out = Conv2D(1, 1, activation='sigmoid')(decoder)

    return Model(inputs=inp, outputs=out)
