import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Lambda

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the VAE model
latent_dim = 2
inputs = Input(shape=(28, 28, 1))
x = Flatten()(inputs)
x = Dense(128, activation='relu')(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

z = Lambda(lambda args: args[0] + args[1]*0.5)([z_mean, z_log_var])

# Decoder
decoder_h = Dense(128, activation='relu')(z)
decoder_mean = Dense(28 * 28, activation='sigmoid')(decoder_h)

# VAE model
vae_output = Reshape((28, 28, 1))(decoder_mean)
vae = Model(inputs, vae_output)
vae.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
vae.fit(x_train, x_train, epochs=10, batch_size=128)

# Display results
decoded_imgs = vae.predict(x_test)
# Display the first decoded image
plt.imshow(decoded_imgs[5].reshape(28, 28), cmap='gray')
plt.show()