#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math
import sys

from matplotlib.gridspec import GridSpec

def get_dataset():

    coherence, frequencies, times = np.load('coherence.npy')
    coherence = coherence.T
    coherence /= coherence.max()
    coherence = 1. - coherence
    design_matrix = coherence[:12000:10, 100:500:10]
    return design_matrix
    
# Get design matrix
X = get_dataset()    
X = np.sqrt(X)
n_times, n_frequencies = X.shape
n_train = 600
X_train = X[:n_train, :]
X_test = X[n_train:, :]
x_shape = [1, n_times, n_frequencies, 1]
X = X.reshape(x_shape)
x_tain_shape = [1, n_train, n_frequencies, 1]
X_train = X_train.reshape(x_tain_shape)
X_test = X_test.reshape(x_tain_shape)

# Define input tensor
x_shape = [1, n_train, n_frequencies, 1]
x = tf.placeholder(tf.float32, x_shape, name='x')

# Learning architecture
training_epochs = 100000
learning_rate = 0.001

# Convolutional encoder
layer_size = 20
filter_dim = 5 
W_var = 1.0/math.sqrt(n_frequencies*n_train)
W_dim = [filter_dim, filter_dim, 1, layer_size]
W_filters = tf.Variable(tf.random_uniform(W_dim, -W_var, W_var)) 
b_filters = tf.Variable(tf.zeros([layer_size]))
ckw = dict(strides=[1, 1, 1, 1], padding='SAME')
latent = tf.nn.relu(tf.nn.conv2d(x, W_filters, **ckw))

to_encode = tf.reshape(latent, [layer_size, n_train*n_frequencies])
W_var = 1.0/math.sqrt(n_frequencies*n_train)

encoding_layers = 4 

W_encode = tf.Variable(tf.random_uniform([n_train*n_frequencies, encoding_layers], 
    -W_var, W_var))

b_encode = tf.Variable(tf.zeros([encoding_layers]))
encoded = tf.nn.tanh(tf.matmul(to_encode, W_encode) + b_encode)

W_decode = tf.transpose(W_encode)
b_decode = tf.Variable(tf.random_uniform([n_train*n_frequencies], 0, 1.))
decode = tf.nn.tanh(tf.matmul(encoded, W_decode) + b_decode)

latent_out = tf.reshape(decode, [1, n_train, n_frequencies, layer_size])

# Deconvolution
decoded = tf.nn.tanh(tf.nn.conv2d_transpose(latent_out, W_filters, x_shape, **ckw))

# Cost & optimizer
cost = tf.reduce_sum(tf.square(decoded - x))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Generate figure
fig = plt.figure(figsize=[15, 8])
grid = GridSpec(3, layer_size)
ax_w = [fig.add_subplot(g) for g in grid]
[a.set_axis_off() for a in ax_w]
ax_w = ax_w[2*layer_size:]

# Show design matrix
imshow_kw = dict(aspect='auto', origin='lower')
ax_x = fig.add_subplot(grid[0, :])
ax_x.imshow(X.reshape(n_times, n_frequencies).T, **imshow_kw)

# Show reconstructed matrix
imshow_kw = dict(aspect='auto', origin='lower', extent=[0, n_times, 0, 1])
ax_r = fig.add_subplot(grid[1, :])
ax_r.imshow(np.zeros((n_times, n_frequencies)).T, **imshow_kw)

fig_latent = plt.figure(figsize=[15, 13])
grid_latent = GridSpec(layer_size, 1) 
ax_latent = [fig_latent.add_subplot(g) for g in grid_latent]
fig_latent.savefig('cnn-layers-latent-fc.pdf', bbox_inches='tight')

# Run graph
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        optimizer.run(feed_dict={x: X_train})   
        print(epoch, sess.run(cost, feed_dict={x:X_train}))

        if (epoch % 1000 == 0):
            
            # Display weights & latent
            weights = sess.run(W_filters, feed_dict={x:X_train})
            weights_encoded = sess.run(W_encode, feed_dict={x:X_train})
            bias_encoded = sess.run(b_encode, feed_dict={x:X_train})
            
            latent_train = sess.run(latent, feed_dict={x:X_train})
            latent_test = sess.run(latent, feed_dict={x:X_test})
            latent_r = np.hstack([latent_train, latent_test])

            for i in range(layer_size):
                ax_w[i].cla()
                ax_w[i].imshow(weights[:, :, 0, i])
                ax_w[i].set_axis_off()
                
                ax_latent[i].imshow(latent_r[..., i].reshape(n_times,
                    n_frequencies).T, **imshow_kw)
            
            

            # Display reconstructed
            rec_train = sess.run(decoded, feed_dict={x:X_train})
            rec_test = sess.run(decoded, feed_dict={x:X_test})
            X_rec = np.hstack([rec_train, rec_test])
            ax_r.cla()
            ax_r.imshow(X_rec.reshape(n_times, n_frequencies).T, **imshow_kw)

            # Save figure and weights
            fig.savefig('cnn-layers-coherence-fc.pdf', bbox_inches='tight')
            fig_latent.savefig('cnn-layers-latent-fc.pdf', bbox_inches='tight')
            print('Display flushed')
            np.save('weights-10', weights)
            np.save('weights-encoded-10', weights_encoded)
            np.save('bias-encoded-10', bias_encoded)
            