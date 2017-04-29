#!/usr/bin/env python
"""
Use VGG16 to extract features from entropogram
"""

from load_data import get_coherence, get_spectrogram
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import scipy
import matplotlib.pyplot as plt


def main():
    model = VGG16(weights='imagenet', include_top=False)
    print(model.input)

    spectrogram, frequency, time = get_spectrogram()
    spectrogram = spectrogram[:, ::5]
    ntimes, nfreqs = spectrogram.shape
    data = np.array([spectrogram, spectrogram, spectrogram])
    data = data.reshape(3, ntimes, nfreqs, 1)
    data = data.transpose((3, 1, 2, 0))
    data = preprocess_input(data)

    model.predict(data)

    feature_maps = []
    for ilayer, layer in enumerate(model.layers):
        feature_model = Model(model.input, layer.output)
        print(layer.name)
        if 'conv' in layer.name:
            fmaps_layer = feature_model.predict(data)

            nsamples, width, height, nchannels = fmaps_layer.shape
            for ifeature in range(nchannels):
                fmap = fmaps_layer[0, :, :, ifeature]
                upscaled = scipy.misc.imresize(fmap, size=(ntimes, nfreqs),
                                               mode="F", interp='bilinear')
                feature_maps.append(upscaled)

    feature_maps = np.array(feature_maps)
    nfeatures, ntimes, nfreqs = feature_maps.shape

    plt.figure()
    plt.plot(np.mean(feature_maps, axis=(1, 2)))

    # kmeans parameters
    from sklearn.cluster import KMeans

    n_clusters = 10

    predictor = KMeans(init='k-means++', n_clusters=n_clusters, n_init=5)
    predictor.fit(feature_maps.reshape(nfeatures, ntimes*nfreqs).T)
    fig, ax = plt.subplots()
    ax.imshow(predictor.labels_.reshape(ntimes, nfreqs).T, aspect='auto',
              origin='lower')

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

if __name__ == "__main__":
    main()
