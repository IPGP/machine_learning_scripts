#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Clean convoutional autoencoder that can be used to test some parametrizations.
"""

import matplotlib.pyplot as plt
import numpy as np

from load_data import get_coherence, get_spectrogram
from keras.layers import Input, Dense, MaxPooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import UpSampling2D, Conv2D, Reshape
from keras.models import Model
from keras.optimizers import Adam


def main():
    # entropy, frequency, time = get_coherence(decimate=3)
    # data = np.log10(np.sqrt(entropy.T))

    spectrogram, frequency, time = get_spectrogram()
    data = spectrogram

    # define filter sizes
    nfilters = [10]
    filter_sizes = [10]
    ndense = 2
    pooling = [(10, 1)]
    nlayers = len(nfilters)
    dec1 = np.prod([dec[0] for dec in pooling])
    dec2 = np.prod([dec[1] for dec in pooling])

    # adapt data size to fit pooling
    ntimes = data.shape[0] // dec1 * dec1
    data = data[:ntimes, ::10]
    x_shape = data.shape
    shape_compressed = x_shape[0]//dec1, x_shape[1]//dec2
    X = data[None, :, :, None]
    print('data shape:', x_shape)
    print('decimation:', dec1, dec2)
    print('data shape compressed:', shape_compressed)

    activation = LeakyReLU(0.05)
    autoencoder = get_model(x_shape, nfilters, filter_sizes, ndense, pooling,
                            activation=activation)
    autoencoder.summary()
    optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                     decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss='mean_absolute_error')
    history = autoencoder.fit(X, X, epochs=5000, verbose=2)

    fig, row1 = plt.subplots(1, 1, sharex=True)
    row1.plot(history.history['loss'])
    row1.set(xlabel='epoch', ylabel='loss')

    decoded_img = autoencoder.predict(X)

    # ----PLOTTING ----
    # input and output data
    layer = autoencoder.layers[-1]
    model = Model(autoencoder.input, layer.output)
    decoded_img = model.predict(X)
    fig, (row1, row2) = plt.subplots(2, 1)
    row1.imshow(X.reshape(x_shape).T, aspect='auto', origin='lower', vmin=0.,
                vmax=X.max())
    row2.imshow(decoded_img.reshape(x_shape).T, aspect='auto', origin='lower',
                vmin=0., vmax=X.max())

    # plot each layer (except maxpooling)
    for ilayer, layer in enumerate(autoencoder.layers):
        print(ilayer, layer.name)
        model = Model(autoencoder.input, layer.output)
        layer_weights = layer.get_weights()
        layer_output = model.predict(X)
        if layer.name[:6] == 'conv2d':
            plot_convlayer(layer_weights, layer_output)
        elif layer.name == 'dense_1':
            plot_denselayer(shape_compressed, layer_weights, layer_output)
    plt.show()


def plot_convlayer(layer_weights, layer_output):
    filter_weights, filter_biases = layer_weights
    size1, size2, chan1, chan2 = filter_weights.shape
    print(filter_weights.shape)
    print(layer_output.shape)
    fig = plt.figure()
    fig.suptitle('conv2d layer')

    ax_old = None
    for ichan2 in range(chan2):
        for ichan1 in range(chan1):
            ax = plt.subplot2grid((chan2, chan1+5), (ichan2, ichan1),
                                  colspan=1)
            ax.imshow(filter_weights[:, :, ichan1, ichan2])

        ax = plt.subplot2grid((chan2, chan1+5), (ichan2, chan1), colspan=5,
                              sharex=ax_old, sharey=ax_old)
        ax.imshow(layer_output[0, :, :, ichan2].T,
                  aspect='auto', origin='lower')
        ax_old = ax


def plot_denselayer(x_shape, layer_weights, layer_output):
    filter_weights, filter_biases = layer_weights
    print(filter_weights.shape)
    print(layer_output.shape)
    _, npixels, ndense = layer_output.shape
    fig, axes = plt.subplots(ndense, 1, sharey=True, sharex=True)
    if ndense == 1:
        axes = [axes]

    fig.suptitle('dense layer')
    for iax, ax in enumerate(axes):
        ax.imshow(layer_output[:, :, iax].reshape(x_shape).T,
                  aspect='auto', origin='lower')


def get_model(x_shape, nfilters, filter_sizes, ndense, pooling,
              activation='tanh'):
    """
    Example architecture for nfilter=[2, 3], filter_sizes=[3, 4], ndense=2


    [16, 16] filter3<1,2> [16, 16, 2] pool [8, 8, 2] filter4<2,3> [8, 8, 3]
    pool [4, 4, 3] dense [16, 3] bottleneck [16, 2] dense [4, 4, 3] unpool [8,
    8, 3] filter4<3,2> [8, 8, 2] unpool [16, 16, 2] filter3<2,1> [16, 16]
    """

    # Input layer
    input_image = Input(shape=(x_shape[0], x_shape[1], 1))

    # Define layers
    nlayers = len(nfilters)
    layers = []
    dec1 = np.prod([dec[0] for dec in pooling])
    dec2 = np.prod([dec[1] for dec in pooling])
    shape_compressed = x_shape[0]//dec1, x_shape[1]//dec2

    # encode
    for nfilter, filter_size, pool in zip(nfilters, filter_sizes, pooling):
        conv = Conv2D(nfilter, filter_size, activation=activation,
                      padding='same')
        maxp = MaxPooling2D(pool, padding='same')
        drop = Dropout(0.1)
        layers.append(drop)
        layers.append(conv)
        layers.append(maxp)

    # dense network with bottleneck
    flat = Reshape((shape_compressed[0] * shape_compressed[1], nfilters[-1]))
    dense = Dense(ndense, activation='sigmoid')
    udense = Dense(nfilters[-1])
    uflat = Reshape((shape_compressed[0], shape_compressed[1], nfilters[-1]))

    layers.append(flat)
    layers.append(dense)
    layers.append(udense)
    layers.append(uflat)

    # decode
    for nfilter, filter_size, pool in zip(nfilters[-2::-1],
                                          filter_sizes[-1:0:-1],
                                          pooling[-1:0:-1]):
        dmaxp = UpSampling2D(pool)
        dconv = Conv2D(nfilter, filter_size, activation=activation,
                       padding='same')
        layers.append(dmaxp)
        layers.append(dconv)

    # output layer
    output_dmaxp = UpSampling2D(pooling[0])
    output_layer = Conv2D(1, filter_sizes[0], activation='tanh',
                          padding='same')

    # --- now connect the model with input ----
    x = input_image
    for layer in layers:
        x = layer(x)
        print(layer.name, '-->', x)
    x = output_dmaxp(x)
    output_image = output_layer(x)
    autoencoder = Model(input_image, output_image)
    return autoencoder


if __name__ == "__main__":
    main()
