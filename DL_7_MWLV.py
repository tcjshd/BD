import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST and filter only digit "0"
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[y_train == 0]
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(-1, 784)

# Architecture parameters
input_dim, hidden_dim, latent_dim = 784, 128, 2

# Encoder weights
W1 = tf.Variable(tf.random.normal([input_dim, hidden_dim], stddev=0.01))
b1 = tf.Variable(tf.zeros([hidden_dim]))
W_mu = tf.Variable(tf.random.normal([hidden_dim, latent_dim], stddev=0.01))
b_mu = tf.Variable(tf.zeros([latent_dim]))
W_logvar = tf.Variable(tf.random.normal([hidden_dim, latent_dim], stddev=0.01))
b_logvar = tf.Variable(tf.zeros([latent_dim]))

# Decoder weights
W2 = tf.Variable(tf.random.normal([latent_dim, hidden_dim], stddev=0.01))
b2 = tf.Variable(tf.zeros([hidden_dim]))
W_out = tf.Variable(tf.random.normal([hidden_dim, input_dim], stddev=0.01))
b_out = tf.Variable(tf.zeros([input_dim]))

# Sampling using reparameterization
def sample(mu, log_var):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps

# Forward pass
def forward(x):
    h = tf.nn.relu(tf.matmul(x, W1) + b1)
    mu = tf.matmul(h, W_mu) + b_mu
    log_var = tf.matmul(h, W_logvar) + b_logvar
    z = sample(mu, log_var)
    h_dec = tf.nn.relu(tf.matmul(z, W2) + b2)
    x_recon = tf.nn.sigmoid(tf.matmul(h_dec, W_out) + b_out)
    return x_recon, mu, log_var

# Loss
def vae_loss(x):
    x_recon, mu, log_var = forward(x)
    recon = -tf.reduce_sum(x * tf.math.log(x_recon + 1e-10) + (1 - x) * tf.math.log(1 - x_recon + 1e-10), axis=1)
    kl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
    return tf.reduce_mean(recon + kl)

# Training loop
optimizer = tf.optimizers.Adam(1e-3)
batch_size, epochs = 64, 10
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(batch_size)

for epoch in range(epochs):
    for batch in dataset:
        with tf.GradientTape() as tape:
            loss = vae_loss(batch)
        vars = [W1, b1, W_mu, b_mu, W_logvar, b_logvar, W2, b2, W_out, b_out]
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

# Generate samples
def generate():
    grid = 10
    image = np.zeros((28*grid, 28*grid))
    for i, yi in enumerate(np.linspace(-2, 2, grid)):
        for j, xi in enumerate(np.linspace(-2, 2, grid)):
            z = np.array([[xi, yi]], dtype=np.float32)
            h = tf.nn.relu(tf.matmul(z, W2) + b2)
            x_dec = tf.nn.sigmoid(tf.matmul(h, W_out) + b_out)
            digit = x_dec.numpy().reshape(28, 28)
            image[i*28:(i+1)*28, j*28:(j+1)*28] = digit
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

generate()
