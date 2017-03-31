#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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
    ntrain = 400

    plot(design_matrix, labels, ntrain)

    # normalize each row of the design matrix
    design_matrix /= design_matrix.max(axis=1)[:, None]

    training_matrix = design_matrix[0:ntrain, :]
    training_labels = labels[0:ntrain]

    # Model definition
    print(training_matrix.shape)
    x = tf.placeholder(tf.float32, [None, nfeatures, 1], name='x')
    y_label = tf.placeholder(tf.float32, [None, 1], name='y_label')

    def conv1d(x, W, b, stride=2):
        x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    nreduced = nfeatures // 2
    W1 = tf.Variable(tf.constant(1., shape=[5, 1, 1]))
    b1 = tf.Variable(tf.zeros([1]))

    cnn = conv1d(x, W1, b1)
    cnn_reshaped = tf.reshape(cnn, [-1, nreduced])

    W2 = tf.Variable(tf.zeros([nreduced, 1]))
    b2 = tf.Variable(tf.zeros([1]))
    y_predicted = tf.matmul(cnn_reshaped, W2) + b2

    cost = tf.reduce_mean(tf.square(y_predicted - y_label))
    optimizer = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)

    init = tf.initialize_all_variables()

    x0 = training_matrix.reshape(ntrain, nfeatures, 1)
    y0 = training_labels.reshape(ntrain, 1)
    feed = {x: design_matrix.reshape(nsamples, nfeatures, 1)}
    with tf.Session() as sess:
        sess.run(init)

        for isample in range(10000):
            optimizer.run(feed_dict={x: x0, y_label: y0})

        pre_labels = sess.run(y_predicted, feed_dict=feed)
        weights1 = sess.run(W1)
        cnn1 = sess.run(cnn, feed_dict=feed)
        weights2 = sess.run(W2)

        writer = tf.train.SummaryWriter('./', sess.graph)
        writer.close()

    # plotting
    fig, ax = plt.subplots()
    #ax.set_xlim((0, nsamples))
    ax.plot(labels, label='label')
    ax.plot(pre_labels, label='predicted label')
    ax.legend()
    #ax.axvline(ntrain, color='red')
    #ax.set_xlabel('example')
    #ax.set_ylabel('labels - predicted labels')
    fig, ax = plt.subplots()
    ax.plot(weights1.reshape(5))
    ax.set_xlabel('coefficient number')
    ax.set_ylabel('coefficient value')

    fig, ax = plt.subplots()
    ax.imshow(cnn1.reshape(nsamples, nreduced), aspect='auto')
    ax.set_xlabel('coefficient number')
    ax.set_ylabel('coefficient value')

    fig, ax = plt.subplots()
    ax.plot(weights2)
    ax.set_xlabel('coefficient number')
    ax.set_ylabel('coefficient value')

    #fig, ax = plt.subplots()
    #levels = np.linspace(-5000, 5000)
    #hist1, bins1 = np.histogram(training_errors, levels, normed=True)
    #center1 = 0.5 * (bins1[:-1] + bins1[1:])
    #hist2, bins2 = np.histogram(test_errors, levels, normed=True)
    #center2 = 0.5 * (bins1[:-1] + bins1[1:])
    #hist3, bins3 = np.histogram(mean_errors, levels, normed=True)
    #center3 = 0.5 * (bins1[:-1] + bins1[1:])
    #ax.plot(center1, hist1, label='training')
    #ax.plot(center2, hist2, label='test')
    #ax.plot(center3, hist3, label='dataset')
    #ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
