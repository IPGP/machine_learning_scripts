#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def main():
    # ---- dataset ----
    npoints = 400
    x0 = np.linspace(0., 10 * np.pi, npoints)
    noise = np.random.normal(scale=0.1, size=npoints)
    y0 = (np.sin(x0) + noise)
    # y0 = noise

    # ---- definitions ----
    # Model
    nnetwork = 400

    x = tf.placeholder(tf.float32, [None, 1], name='x')
    y_label = tf.placeholder(tf.float32, [None, 1], name='y')

    W1 = tf.Variable(tf.random_uniform([1, nnetwork], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([nnetwork]))

    h1 = tf.nn.tanh(tf.matmul(x, W1) + b1)


    W2 = tf.Variable(tf.random_uniform([nnetwork, 1], -nnetwork, nnetwork))
    b2 = tf.Variable(tf.zeros([1]))

    y = tf.matmul(h1, W2) + b2

    # cost function
    cost = tf.reduce_mean(tf.square(y_label - y))

    # optimizer
    stepsize = 0.001
    optimizer = tf.train.GradientDescentOptimizer(stepsize).minimize(cost)

    # ---- run tensorflow ----
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for isample in range(10000):
            optimizer.run(feed_dict={x: x0[:, None], y_label: y0[:, None]})

        y_predicted = sess.run(y, feed_dict={x: x0[:, None]})

        # write graph file
        writer = tf.train.SummaryWriter('./', sess.graph)
        writer.close()

    # ---- plotting ----
    fig, axes = plt.subplots()
    axes.plot(x0, y0, 'o')
    axes.plot(x0, y_predicted)

    plt.show()


if __name__ == "__main__":
    main()
