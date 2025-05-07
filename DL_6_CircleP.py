import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Generate a circular dataset
n_samples = 1000
noise = 0.1
dataset, labels = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)

# Visualize the circular data
plt.figure(figsize=(8, 8))
plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap=plt.cm.Paired)
plt.title("Circular Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')
plt.show()

# PCA to reduce dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(dataset)

# Inverse transform the data
inverse_transformed_data = pca.inverse_transform(principal_components)

# Calculate Mean Squared Error for PCA and Inverse PCA
mse_pca = mean_squared_error(dataset, inverse_transformed_data)
print("Mean Squared Error for PCA & Inverse PCA is ", mse_pca)

# Autoencoder using MLPRegressor
autoencoder = MLPRegressor(hidden_layer_sizes=(2024, 2, 2024), activation='relu', solver='adam', max_iter=100000000)

autoencoder.fit(dataset, dataset)

# Reconstructed data
reconstructed_data = autoencoder.predict(dataset)

# Mean Squared Error for Autoencoder
mse_autoencoder = mean_squared_error(dataset, reconstructed_data)

print("Mean Squared Error for Autoencoder is ", mse_autoencoder)
