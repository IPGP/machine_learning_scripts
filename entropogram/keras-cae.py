#!/usr/bin/env python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

from coherence import get_coherence
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

# Get design matrix
X, _, _ = get_coherence(decimate=3)    
X = np.sqrt(X.T)
X = X[:3994, :50]
# plt.figure(figsize=[10, 1])
# plt.imshow(X.T, aspect='auto', origin='lower')
# plt.savefig('design-matrix.pdf')
# exit()

x_shape = X.shape
X = X.reshape([1, X.shape[0], X.shape[1], 1])


input_img = Input(shape=(x_shape[0], x_shape[1], 1))  

conv1 = Conv2D(3, (8, 8), activation='tanh', padding='same')
maxp1 = MaxPooling2D((2, 2), padding='same')
conv2 = Conv2D(8, (4, 4), activation='tanh', padding='same')
maxp2 = MaxPooling2D((2, 2), padding='same')
conv3 = Conv2D(8, (4, 4), activation='tanh', padding='same')
maxp3 = MaxPooling2D((2, 2), padding='same')

dconv4 = Conv2D(8, (4, 4), activation='tanh', padding='same')
dmaxp3 = UpSampling2D((2, 2))
dconv3 = Conv2D(8, (4, 4), activation='tanh', padding='same')
dmaxp2 = UpSampling2D((2, 2))
dconv2 = Conv2D(3, (4, 4), activation='tanh')
dmaxp1 = UpSampling2D((2, 2))
dconv1 = Conv2D(1, (8, 8), activation='tanh', padding='same')

x = conv1(input_img)
x = maxp1(x)
x = conv2(x)
x = maxp2(x)
x = conv3(x)
encoded = maxp3(x)
x = dconv4(encoded)
x = dmaxp3(x)
x = dconv3(x)
x = dmaxp2(x)
x = dconv2(x)
x = dmaxp1(x)
decoded = dconv1(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadetla', loss='mean_squared_error')
autoencoder.fit(X, X, epochs=200)



# plt.figure()
# decoded_img = autoencoder.predict(X)
# decoded_img = decoded_img.reshape(x_shape)
# plt.imshow(decoded_img.T, aspect='auto', origin='lower')
# plt.show()

conv1 = Conv2D(3, (8, 8), activation='sigmoid', padding='same', 
	weights=autoencoder.layers[1].get_weights())
maxp1 = MaxPooling2D((2, 2), padding='same')
conv2 = Conv2D(8, (4, 4), activation='sigmoid', padding='same', 
	weights=autoencoder.layers[3].get_weights())
maxp2 = MaxPooling2D((2, 2), padding='same')
conv3 = Conv2D(8, (4, 4), activation='sigmoid', padding='same', 
	weights=autoencoder.layers[5].get_weights())
maxp3 = MaxPooling2D((2, 2), padding='same')

dconv4 = Conv2D(8, (4, 4), activation='sigmoid', padding='same', 
	weights=autoencoder.layers[7].get_weights())
dmaxp3 = UpSampling2D((2, 2))
dconv3 = Conv2D(8, (4, 4), activation='sigmoid', padding='same', 
	weights=autoencoder.layers[9].get_weights())
dmaxp2 = UpSampling2D((2, 2))
dconv2 = Conv2D(3, (4, 4), activation='sigmoid', 
	weights=autoencoder.layers[11].get_weights())
dmaxp1 = UpSampling2D((2, 2))

x = conv1(input_img)
x = maxp1(x)
x = conv2(x)
x = maxp2(x)
x = conv3(x)
encoded = maxp3(x)
x = dconv4(encoded)
x = dmaxp3(x)
x = dconv3(x)
x = dmaxp2(x)
x = dconv2(x)
almost = dmaxp1(x)


seener = Model(input_img, almost)
decoded_img = seener.predict(X)
print(decoded_img.shape)
plt.figure(figsize=(10, 4))

for i in range(3):
	plt.subplot(3+1, 1, i+1)
	plt.imshow(decoded_img[..., i].reshape(x_shape).T, aspect='auto', 
		origin='lower')

plt.subplot(4, 1, 4)
plt.imshow(X.reshape(x_shape).T, aspect='auto', origin='lower')
plt.show()

