# %pip install scikit-learn
# %pip install matplotlib

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load and preprocess dog & cat images
img_size = (64, 64)  # You can change this if needed
batch_size = 200  # Keep it small for memory efficiency

datagen = ImageDataGenerator(rescale=1./255)

# Load images from folder
base_dir = 'PetImages'  # Folder containing Cat and Dog subfolders
generator = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# 2. Extract one batch of images
X_data, _ = next(generator)  # shape: (batch_size, 64, 64, 3)

# Flatten and standardize
X_flat = X_data.reshape((X_data.shape[0], -1))  # shape: (batch_size, 64*64*3)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# 3. PCA for dimensionality reduction and reconstruction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
inverse_transform_data = pca.inverse_transform(principal_components)
mse_pca = mean_squared_error(X_scaled, inverse_transform_data)
print("✅ Mean Squared Error for PCA & Inverse PCA:", mse_pca)

# 4. Autoencoder using MLP
autoencoder = MLPRegressor(hidden_layer_sizes=(1024, 2, 1024),
                           activation='relu',
                           solver='adam',
                           max_iter=200,
                           verbose=True)

autoencoder.fit(X_scaled, X_scaled)
reconstructed_data = autoencoder.predict(X_scaled)
mse_auto = mean_squared_error(X_scaled, reconstructed_data)
print("✅ Mean Squared Error for Autoencoder:", mse_auto)

# 5. Optional: Visual comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X_data[0])
plt.title("Original Image")

plt.subplot(1, 2, 2)
reconstructed_img = scaler.inverse_transform(reconstructed_data)[0]
reconstructed_img = reconstructed_img.reshape((64, 64, 3))
plt.imshow(np.clip(reconstructed_img, 0, 1))
plt.title("Autoencoder Reconstructed")
plt.tight_layout()
plt.show()
