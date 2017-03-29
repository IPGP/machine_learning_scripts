#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


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
    ntrain = 200

    # label indices:
    # 0: year, 1: month, 2:day, 3:hour, 4:min, 5:sec, 6:lat, 7:lon
    # 8: depth, 9: M0, 10: Mw, 11:strike1, 12: dip1, 13: rake1, 14:strike2,
    # 15:dip2, 16:rake2
    plot(design_matrix, labels, ntrain)

    # normalize design matrix
    design_matrix /= design_matrix.max(axis=1)[:, None]
    #design_matrix -= design_matrix.mean()

    # extract training data
    training_matrix = design_matrix[0:ntrain, :]

    # Model definition
    x = tf.placeholder(tf.float32, [None, nfeatures], name='x')
    y_label = tf.placeholder(tf.float32, [None, nfeatures], name='y_label')

    layer_sizes = [4, 1]
    nlayers = len(layer_sizes)
    autoencoder = create(x, layer_sizes)

    optimizer = tf.train.MomentumOptimizer(0.1, 0.001).minimize(autoencoder['cost'])

    init = tf.initialize_all_variables()

    x0 = training_matrix.reshape(ntrain, nfeatures)

    # plot original and reconstructed data
    fig, (col1, col2) = plt.subplots(1, 2, sharex=True, sharey=True)
    col1.imshow(x0, aspect='auto', vmin=0., vmax=1.)
    image = col2.imshow(np.zeros_like(x0), aspect='auto', vmin=0., vmax=1.)
    fig.show()
    plt.pause(0.01)

    with tf.Session() as sess:
        sess.run(init)

        for istep in range(200000):
            o, c = sess.run([optimizer, autoencoder['cost']], feed_dict={x: x0,
                            y_label: x0})

            if (istep % 1000 == 0):
                print('Loss at step {}: {}'.format(istep, c))
                pre_labels = sess.run(autoencoder['decoded'], feed_dict={x: x0})
                image.set_data(pre_labels)
                plt.draw()
                plt.pause(0.01)

        pre_labels = sess.run(autoencoder['decoded'], feed_dict={x: x0})
        layer1_calc = sess.run(autoencoder['encoded'], feed_dict={x: x0})
        weights = sess.run(autoencoder['weights'])

        writer = tf.train.SummaryWriter('./', sess.graph)
        writer.close()

    # plot encoded data
    fig, (col1, col2) = plt.subplots(1, 2, sharey=True)
    col1.imshow(x0, aspect='auto')
    for icoeff, coeffs in enumerate(layer1_calc.T):
        col2.plot(coeffs, range(ntrain))

    # plot weights
    fig, axes = plt.subplots(nlayers, 1)
    if nlayers == 1:
        axes = [axes]
    for icoeff, coeffs in enumerate(weights):
        for iline, line in enumerate(coeffs.T):
            axes[icoeff].plot(line)

    plt.show()


def create(x, layer_sizes):
    # Build the encoding layers
    next_layer_input = x
    activation = [tf.nn.sigmoid, tf.nn.tanh]

    encoding_matrices = []
    for ilayer, dim in enumerate(layer_sizes):
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(tf.random_uniform([input_dim, dim],
            -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
        # Initialize b to zero
        b = tf.Variable(tf.random_uniform([dim], -1, 1.))

        # We are going to use tied-weights so store the W matrix for later
        # reference.
        encoding_matrices.append(W)

        output = activation[ilayer](tf.matmul(next_layer_input, W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()
    activation.reverse()

    for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]):
        # we are using tied weights, so just lookup the encoding matrix for
        # this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.random_uniform([dim], -1, 1.))
        output = activation[i](tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input
    lsq_error = tf.sqrt(tf.reduce_mean(tf.square(x - reconstructed_x)))
    cost = lsq_error

    #alpha = 1.
    #for weights in encoding_matrices[1:]:
    #    constraint = -tf.minimum(tf.reduce_min(weights), 0)
    #    cost += alpha * constraint

    return {
           'weights': encoding_matrices,
           'encoded': encoded_x,
           'decoded': reconstructed_x,
           'cost': cost
           }


if __name__ == "__main__":
    main()
