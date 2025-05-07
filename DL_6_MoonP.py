import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

n_samples = 1000
noise = 0.1
dataset, labels = make_moons(n_samples=n_samples, noise=noise, random_state=42)

plt.figure(figsize=(8, 8))
plt.scatter(dataset[:, 0], dataset[:, 1])

plt.title("Non-Symmetric Data (Crescent Shape)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')
plt.show()

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

pca = PCA(n_components=2)
principal_components = pca.fit_transform(dataset)

inverse_transformed_data = pca.inverse_transform(principal_components)

mse = mean_squared_error(dataset, inverse_transformed_data)
print("Mean Squared Error for PCA & Inverse PCA is ", mse)

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

autoencoder = MLPRegressor(hidden_layer_sizes=(2024, 2, 2024), activation='relu', solver='adam', max_iter=100000000)

autoencoder.fit(dataset, dataset)

reconstructed_data = autoencoder.predict(dataset)

mse = mean_squared_error(dataset, reconstructed_data)

print("Mean Squared Error for Autoencoder is ", mse)