#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


def main():
    design_matrix, frequencies, times = np.load('coherence.npy',
                                                encoding='bytes')
    design_matrix = design_matrix.T
    fmax = 0.2
    mask = frequencies < fmax
    design_matrix = design_matrix[:, mask]
    design_matrix = design_matrix[:200, ::9]
    ntimes, nfreqs = design_matrix.shape
    print(ntimes, nfreqs)

    # normalize design matrix
    design_matrix /= design_matrix.max()
    design_matrix = 1. - design_matrix

    # Model definition
    x0 = design_matrix
    x = tf.placeholder(tf.float32, shape=[1, ntimes, nfreqs, 1])

    # Step 1: Convolution
    # for every point of the image, we will get a number for every feature
    # that appears in this environment
    nfeatures = 15
    w = 15
    w1 = tf.Variable(tf.random_uniform([w, w, 1, nfeatures], -1, 1))
    encoded = tf.nn.tanh(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1],
                                      padding='SAME'))

    # Step 2: Deconvolution
    w2 = tf.Variable(tf.random_uniform([w, w, nfeatures, 1], -1, 1))
    decoded = tf.nn.tanh(tf.nn.conv2d(encoded, w2, strides=[1, 1, 1, 1],
                                      padding='SAME'))

    lsq_error = tf.sqrt(tf.reduce_mean(tf.square(x - decoded)))
    cost = lsq_error

    # chose optimizer
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    # initialize variables
    init = tf.initialize_all_variables()

    # plot original and reconstructed data
    plt.ion()

    fig = plt.figure()
    ax00 = plt.subplot2grid((4, 8), (0, 0), rowspan=3)
    ax01 = plt.subplot2grid((4, 8), (0, 1), rowspan=3, sharex=ax00, sharey=ax00)

    ax10 = plt.subplot2grid((4, 3), (3, 0))
    ax11 = plt.subplot2grid((4, 3), (3, 1))
    ax12 = plt.subplot2grid((4, 3), (3, 2))

    ax_features = []
    ncols = 6
    for ifeature in range(nfeatures):
        irow = ifeature // ncols
        icol = 2 + ifeature - irow * ncols
        ax = plt.subplot2grid((4, 8), (irow, icol))
        ax_features.append(ax)


    fig.show()
    plt.pause(0.01)

    ax00.imshow(x0, vmin=0., vmax=1.)
    im_decoded = ax01.imshow(np.zeros_like(x0), vmin=0., vmax=1.)

    nsteps = 100000
    cost_history = np.zeros(nsteps)

    with tf.Session() as sess:
        sess.run(init)

        for istep in range(nsteps):
            o, c = sess.run([optimizer, cost],
                    feed_dict={x: x0[None, :, :, None]})
            cost_history[istep] = c

            if (istep % 100 == 0):
                print('Loss at step {}: {}'.format(istep, c))
                pre_labels = sess.run(decoded, feed_dict={x: x0[None, :, :, None]})
                feature_map = sess.run(w1)

                im_decoded.set_data(pre_labels[0, :, :, 0])

                for iax, ax in enumerate(ax_features):
                    ax.cla()
                    ax.imshow(feature_map[:, :, 0, iax])

                ax10.cla()
                ax10.plot(cost_history[:istep], c='C0')

                plt.draw()
                plt.pause(0.01)


if __name__ == "__main__":
    main()
