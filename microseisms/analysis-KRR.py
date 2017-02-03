#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, kernel_ridge


def plot(matrix, labels, ntrain=None):
    nsamples, nfeatures = matrix.shape
    fig, (col1, col2) = plt.subplots(1, 2, sharey=True)
    col1.autoscale(False)
    col1.set_xlim((0, nfeatures))
    col1.set_ylim((0, nsamples))
    matrix_normed = matrix / np.max(matrix, axis=1)[:, None]
    col1.imshow(matrix_normed, origin='lower', cmap='viridis', aspect='auto')
    col2.plot(labels, range(nsamples))
    if ntrain is not None:
        col1.axhline(ntrain, color='white', linewidth=2, alpha=0.5)
        col2.axhline(ntrain, color='red')
    col1.set_xlabel('features')
    col1.set_ylabel('examples')
    col2.set_xlabel('label')


def main():
    dataset = np.load('dataset.npz')
    design_matrix = dataset['design_matrix']
    labels = dataset['depths']

    mask = np.all(np.isfinite(design_matrix), axis=1)
    design_matrix = design_matrix[mask]
    labels = labels[mask]

    nsamples, nfeatures = design_matrix.shape
    print(nsamples, nfeatures)

    # Number of examples to use for training
    # ntrain = 1000
    ntrain = 50

    # label indices:
    # 0: year, 1: month, 2:day, 3:hour, 4:min, 5:sec, 6:lat, 7:lon
    # 8: depth, 9: M0, 10: Mw, 11:strike1, 12: dip1, 13: rake1, 14:strike2,
    # 15:dip2, 16:rake2
    plot(design_matrix, labels, ntrain)

    # normalize each row of the design matrix
    design_matrix /= design_matrix.max(axis=1)[:, None]

    training_matrix = design_matrix[0:ntrain, :]
    training_labels = labels[0:ntrain]

    # Model definition
    damping_value = 0.1
    reg = kernel_ridge.KernelRidge(alpha=damping_value, kernel='rbf',
                                   gamma=0.5)
    reg.fit(training_matrix, training_labels)

    predicted_labels = reg.predict(design_matrix)
    errors = labels - predicted_labels
    mean_errors = labels - labels.mean()
    training_errors = errors[0:ntrain]
    test_errors = errors[ntrain:]

    fig, ax = plt.subplots()
    ax.set_xlim((0, nsamples))
    ax.plot(errors)
    ax.axvline(ntrain, color='red')
    ax.set_xlabel('example')
    ax.set_ylabel('labels - predicted labels')

    fig, ax = plt.subplots()
    ax.plot(reg.dual_coef_)
    ax.set_xlabel('coefficient number')
    ax.set_ylabel('coefficient value')

    fig, ax = plt.subplots()
    levels = np.linspace(-5000, 5000)
    hist1, bins1 = np.histogram(training_errors, levels, normed=True)
    center1 = 0.5 * (bins1[:-1] + bins1[1:])
    hist2, bins2 = np.histogram(test_errors, levels, normed=True)
    center2 = 0.5 * (bins1[:-1] + bins1[1:])
    hist3, bins3 = np.histogram(mean_errors, levels, normed=True)
    center3 = 0.5 * (bins1[:-1] + bins1[1:])
    ax.plot(center1, hist1, label='training')
    ax.plot(center2, hist2, label='test')
    ax.plot(center3, hist3, label='dataset')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
