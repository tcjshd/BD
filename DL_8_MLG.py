from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load data
(x_train, _), _ = mnist.load_data()
x_train = x_train.astype("float32") / 255.0 

# Generator
generator = Sequential([
    Dense(128, input_dim=100),
    LeakyReLU(0.2),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN model
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
epochs = 10000
batch_size = 64

for epoch in range(epochs):
    # Select random real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]

    # Generate fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_imgs = generator.predict(noise, verbose=0)

    # Train discriminator
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)) * 0.9)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))

    # Train generator via GAN
    noise = np.random.normal(0, 1, (batch_size, 100))
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print losses occasionally
    if epoch % 1000 == 0:
        d_loss_total = d_loss_real[0] + d_loss_fake[0]
        print(f"Epoch {epoch}, D Loss: {d_loss_total:.4f}, G Loss: {g_loss:.4f}")

# Generate and visualize final images
noise = np.random.normal(0, 1, (10, 100))
generated_imgs = generator.predict(noise, verbose=0)

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(generated_imgs[i], cmap='gray')
    plt.axis('off')
plt.show()