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
    design_matrix = design_matrix[:, ::9]
    nsamples, nfeatures = design_matrix.shape
    print(nsamples, nfeatures)

    # Number of examples to use for training
    ntrain = 500

    # normalize design matrix
    design_matrix /= design_matrix.max()
    design_matrix = 1. - design_matrix

    # extract training data
    training_matrix = design_matrix[0:ntrain, :]
    x0 = training_matrix.reshape(ntrain, nfeatures)

    # Model definition
    x = tf.placeholder(tf.float32, [None, nfeatures], name='x')
    y_label = tf.placeholder(tf.float32, [None, nfeatures], name='y_label')

    layer_sizes = [4, 1]
    autoencoder = create(x, layer_sizes)

    optimizer = tf.train.AdamOptimizer(0.0001).minimize(autoencoder['cost'])

    init = tf.initialize_all_variables()

    # plot original and reconstructed data
    plt.ion()

    fig1 = plt.figure()
    col1 = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
    col2 = plt.subplot2grid((4, 2), (0, 1), rowspan=3, sharex=col1, sharey=col1)
    col2a = col2.twiny()

    col11 = plt.subplot2grid((4, 3), (3, 0))
    col21 = plt.subplot2grid((4, 3), (3, 1))
    col22 = plt.subplot2grid((4, 3), (3, 2))

    axes = [col11, col21, col22]
    col1.imshow(x0, aspect='auto', vmin=0., vmax=1.)
    image = col2.imshow(np.zeros_like(x0), aspect='auto', vmin=0., vmax=1.)
    fig1.show()
    plt.pause(0.01)
    nsteps = 200000
    cost_history = np.zeros(nsteps)

    with tf.Session() as sess:
        sess.run(init)

        for istep in range(nsteps):
            o, c = sess.run([optimizer, autoencoder['cost']], feed_dict={x: x0,
                            y_label: x0})
            cost_history[istep] = c

            if (istep % 1000 == 0):
                print('Loss at step {}: {}'.format(istep, c))
                pre_labels = sess.run(autoencoder['decoded'],
                                      feed_dict={x: x0})
                layer1_calc = sess.run(autoencoder['encoded'], feed_dict={x: x0})
                weights = sess.run(autoencoder['weights'])
                image.set_data(pre_labels)

                # plot input weights
                for ax in axes:
                    ax.cla()
                for iline, line in enumerate(weights[-1].T):
                    col11.plot(line)
                col11.set_xlim(-0.5, nfeatures-0.5)

                # plot cost history
                col21.set(yscale='log')
                col21.plot(cost_history[:istep])

                # plot encoded values over the image
                col2a.cla()
                for icoeff, coeffs in enumerate(layer1_calc.T):
                    col2a.step(coeffs, range(ntrain), where='mid', alpha=0.5)

                # plot one-hot encoder values
                col22.cla()
                for icoeff, coeffs in enumerate(layer1_calc.T):
                    encoded = np.zeros((3, layer_sizes[0]))
                    encoded[:, icoeff] = [coeffs.min(),
                                          0.5*(coeffs.min() + coeffs.max()),
                                          coeffs.max()]
                    feature = sess.run(
                            autoencoder['decoded'],
                            feed_dict={autoencoder['encoded']: encoded})
                    col22.plot(feature.T, color='C{:d}'.format(icoeff))

                plt.draw()
                plt.pause(0.01)

        pre_labels = sess.run(autoencoder['decoded'], feed_dict={x: x0})

        writer = tf.train.SummaryWriter('./', sess.graph)
        writer.close()

    # plot encoded data
    fig, (col1, col2) = plt.subplots(1, 2, sharey=True)
    col1.imshow(x0, aspect='auto')

    plt.pause(3600)
    plt.show()


def create(x, layer_sizes):
    # Build the encoding layers
    next_layer_input = x
    activation = [tf.nn.relu, tf.nn.tanh]

    encoding_matrices = []
    for ilayer, dim in enumerate(layer_sizes):
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        vinit = 1.0 / math.sqrt(input_dim)
        W = tf.Variable(tf.random_uniform([input_dim, dim], -vinit, vinit))

        # Initialize b to zero
        b = tf.Variable(tf.random_uniform([dim], 0, vinit))

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
        b = tf.Variable(tf.random_uniform([dim], 0, 1.))
        output = activation[i](tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input
    lsq_error = tf.sqrt(tf.reduce_mean(tf.square(x - reconstructed_x)))
    cost = lsq_error

    # rho = -1 + 2. / layer_sizes[0]
    # h = tf.tanh(encoded_x)
    # regularization = tf.square(rho - tf.reduce_mean(h))
    # beta = 1.
    # cost += beta * regularization

    #alpha = 1.
    #for weights in encoding_matrices[:]:
    #    constraint = - tf.minimum(tf.reduce_min(weights), 0.)
    #    cost += alpha * constraint

    return {
           'weights': encoding_matrices,
           'encoded': encoded_x,
           'decoded': reconstructed_x,
           'cost': cost
           }


if __name__ == "__main__":
    main()
