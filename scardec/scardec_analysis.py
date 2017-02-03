#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def plot(matrix, labels, ntrain=None):
    nsamples, nfeatures = matrix.shape
    fig, (col1, col2) = plt.subplots(1, 2, sharey=True)
    col1.autoscale(False)
    col2.autoscale(False)
    col1.set_xlim((0, nfeatures))
    col1.set_ylim((0, nsamples))
    matrix_normed = matrix / np.max(matrix, axis=1)[:, None]
    col1.imshow(matrix_normed, origin='lower', cmap='viridis', aspect='auto')
    col2.plot(labels[:, 8] / labels[:, 8].max(), range(nsamples))
    col2.plot(labels[:, 10] / labels[:, 10].max(), range(nsamples))
    if ntrain is not None:
        col1.axhline(ntrain, color='white', linewidth=2, alpha=0.5)
        col2.axhline(ntrain, color='red')
    col1.set_xlabel('features')
    col1.set_ylabel('examples')
    col2.set_xlabel('label')


def main():
    dataset = np.load('dataset.npz')
    design_matrix = dataset['source_time_functions']
    # Decimate the design_matrix, using one feature every ten
    design_matrix = design_matrix[:, ::10]
    nsamples, nfeatures = design_matrix.shape
    print(nsamples, nfeatures)
    labels = dataset['labels']

    # Number of examples to use for training
    # ntrain = 1000
    ntrain = 500

    # label indices:
    # 0: year, 1: month, 2:day, 3:hour, 4:min, 5:sec, 6:lat, 7:lon
    # 8: depth, 9: M0, 10: Mw, 11:strike1, 12: dip1, 13: rake1, 14:strike2,
    # 15:dip2, 16:rake2
    plot(design_matrix, labels, ntrain)

    # normalize each row of the design matrix
    design_matrix /= design_matrix.max(axis=1)[:, None]

    all_mag = labels[:, 10]

    training_matrix = design_matrix[0:ntrain, :]
    training_mag = all_mag[0:ntrain]

    reg = linear_model.LinearRegression()
    reg.fit(training_matrix, training_mag)

    predicted_mag = reg.predict(design_matrix)
    fig, ax = plt.subplots()
    ax.set_xlim((0, nsamples))
    ax.plot(all_mag-predicted_mag)
    ax.axvline(ntrain, color='red')
    ax.set_xlabel('example')
    ax.set_ylabel('delta mag')

    fig, ax = plt.subplots()
    ax.plot(reg.coef_)
    ax.set_xlabel('coefficient number')
    ax.set_ylabel('coefficient value')

    plt.show()


if __name__ == "__main__":
    main()
