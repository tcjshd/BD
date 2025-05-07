import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Create a randomly distributed dataset
def make_random_distribution(n_samples=1000, noise=0.1, random_state=None):
    np.random.seed(random_state)
    
    # Generate random points in a 2D space
    # We'll create some structure by combining multiple distributions
    
    # First distribution: Cluster in the center
    center_count = int(n_samples * 0.4)
    center_x = np.random.normal(0, 0.5, center_count)
    center_y = np.random.normal(0, 0.5, center_count)
    center_points = np.column_stack((center_x, center_y))
    
    # Second distribution: Uniform distribution in the rectangle
    uniform_count = int(n_samples * 0.3)
    uniform_x = np.random.uniform(-2, 2, uniform_count)
    uniform_y = np.random.uniform(-2, 2, uniform_count)
    uniform_points = np.column_stack((uniform_x, uniform_y))
    
    # Third distribution: Ring-like distribution
    ring_count = n_samples - center_count - uniform_count
    radius = 3.0
    theta = np.random.uniform(0, 2*np.pi, ring_count)
    r = radius + np.random.normal(0, 0.3, ring_count)
    ring_x = r * np.cos(theta)
    ring_y = r * np.sin(theta)
    ring_points = np.column_stack((ring_x, ring_y))
    
    # Combine all distributions
    points = np.vstack((center_points, uniform_points, ring_points))
    
    # Shuffle the points
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    points = points[indices]
    
    # Create dummy labels (not used but kept for compatibility)
    labels = np.zeros(n_samples)
    
    return points, labels

# Generate the random dataset
n_samples = 1000
noise = 0.1
dataset, labels = make_random_distribution(n_samples=n_samples, noise=noise, random_state=42)

# Plot the dataset
plt.figure(figsize=(8, 8))
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.title("Randomly Distributed Dataset")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')
plt.show()

# PCA part - unchanged
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(dataset)
inverse_transformed_data = pca.inverse_transform(principal_components)
mse = mean_squared_error(dataset, inverse_transformed_data)
print("Mean Squared Error for PCA & Inverse PCA is ", mse)

# Autoencoder part - unchanged
from sklearn.neural_network import MLPRegressor
autoencoder = MLPRegressor(hidden_layer_sizes=(2024, 2, 2024), activation='relu', 
                          solver='adam', max_iter=100000000)
autoencoder.fit(dataset, dataset)
reconstructed_data = autoencoder.predict(dataset)
mse = mean_squared_error(dataset, reconstructed_data)
print("Mean Squared Error for Autoencoder is ", mse)

# Visualize the original vs reconstructed data
plt.figure(figsize=(15, 7))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.5)
plt.title("Original Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')

# Reconstructed data
plt.subplot(1, 2, 2)
plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], alpha=0.5, color='red')
plt.title("Reconstructed Data (Autoencoder)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')

plt.tight_layout()
plt.show()
