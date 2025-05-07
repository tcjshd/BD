from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, _), _ = mnist.load_data()
X_data = X_train[:10000]  # reduce if memory issue

# Flatten and normalize
X_flat = X_data.reshape((X_data.shape[0], -1)) / 255.0
X_scaled = StandardScaler().fit_transform(X_flat)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
inverse_transform_data = pca.inverse_transform(principal_components)
mse_pca = mean_squared_error(X_scaled, inverse_transform_data)
print("Mean Squared Error for PCA & Inverse PCA is", mse_pca)
autoencoder = MLPRegressor(hidden_layer_sizes=(784, 2, 784),activation='relu',solver='adam',max_iter=10)
autoencoder.fit(X_scaled, X_scaled)
reconstructed_data = autoencoder.predict(X_scaled)
mse_auto = mean_squared_error(X_scaled, reconstructed_data)
print("Mean Squared Error for Autoencoder is", mse_auto)