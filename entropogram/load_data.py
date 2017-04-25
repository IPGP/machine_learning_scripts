#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt


def get_coherence(decimate=None):
    coherence, frequencies, times = np.load('coherence.npy')
    coherence /= coherence.max()
    coherence = 1. - coherence

    if decimate is not None:
        coherence = coherence[150:300:3*decimate, :12000:decimate]
    return coherence, frequencies, times


def plot_coherence():
    coherence, frequencies, times = get_coherence(decimate=3)
    coherence = np.log10(coherence)
    coherence = coherence[:50, :3994]

    fig, ax = plt.subplots(1, figsize=[10, 1])
    ax.imshow(coherence, origin='lower', aspect='auto')
    ax.set_axis_off()

    plt.savefig('show-coherence.pdf', bbox_inches='tight')


def get_spectrogram(decimate=None):
    from scipy.io import loadmat
    data = loadmat('TAM.LHZ.2014_acc.mat')
    spectrogram = data['spectre0']
    isnan = np.any(np.isnan(spectrogram), axis=1)

    spectrogram = np.log10(spectrogram[~isnan, :200])
    spectrogram -= spectrogram.min()
    spectrogram /= spectrogram.max()

    #plt.figure()
    #im = plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    #plt.colorbar(im)
    #plt.show()

    times = data['date0']
    frequencies = data['frq0']
    return spectrogram, frequencies, times


if __name__ == '__main__':
    plot_coherence()

