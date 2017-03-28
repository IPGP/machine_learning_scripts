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
    y0 = np.sin(x0) + noise
    # y0 = noise

    # ---- definitions ----
    # Model
    x = tf.placeholder(tf.float32, [None], name='x')
    y_label = tf.placeholder(tf.float32, [None], name='y')

    m = tf.Variable(tf.ones([1]))
    b = tf.Variable(tf.ones([1]))

    y = m * x + b

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
            optimizer.run(feed_dict={x: x0, y_label: y0})

        y_predicted = sess.run(y, feed_dict={x: x0})
        m_optimized = sess.run(m)
        b_optimized = sess.run(b)

        # write graph file
        writer = tf.train.SummaryWriter('./', sess.graph)
        writer.close()

    print(m_optimized, b_optimized)
    # ---- plotting ----
    fig, axes = plt.subplots()
    axes.plot(x0, y0)
    axes.plot(x0, y_predicted)

    plt.show()


if __name__ == "__main__":
    main()
