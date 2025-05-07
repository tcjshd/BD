import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Create a rectangle-shaped dataset instead of moons
def make_rectangle(n_samples=1000, noise=0.1, random_state=None):
    np.random.seed(random_state)
    
    # Generate points along the perimeter of a rectangle
    width, height = 2.0, 1.0
    
    # Calculate how many points to put on each side
    points_per_side = [int(n_samples * width / (2 * (width + height))),  # bottom
                       int(n_samples * height / (2 * (width + height))),  # right
                       int(n_samples * width / (2 * (width + height))),   # top
                       int(n_samples * height / (2 * (width + height)))]   # left
    
    # Adjust the last value to ensure we get exactly n_samples points
    points_per_side[3] += n_samples - sum(points_per_side)
    
    # Create the rectangle perimeter points
    points = []
    
    # Bottom edge (x varies, y=0)
    x_bottom = np.linspace(-width/2, width/2, points_per_side[0])
    y_bottom = np.zeros(points_per_side[0])
    bottom_edge = np.column_stack((x_bottom, y_bottom))
    
    # Right edge (x=width/2, y varies)
    x_right = np.ones(points_per_side[1]) * (width/2)
    y_right = np.linspace(0, height, points_per_side[1])
    right_edge = np.column_stack((x_right, y_right))
    
    # Top edge (x varies, y=height)
    x_top = np.linspace(width/2, -width/2, points_per_side[2])
    y_top = np.ones(points_per_side[2]) * height
    top_edge = np.column_stack((x_top, y_top))
    
    # Left edge (x=-width/2, y varies)
    x_left = np.ones(points_per_side[3]) * (-width/2)
    y_left = np.linspace(height, 0, points_per_side[3])
    left_edge = np.column_stack((x_left, y_left))
    
    # Combine all edges
    points = np.vstack((bottom_edge, right_edge, top_edge, left_edge))
    
    # Add noise
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
    
    # Create dummy labels (not used but kept for compatibility)
    labels = np.zeros(n_samples)
    
    return points, labels

# Generate the rectangle dataset
n_samples = 1000
noise = 0.1
dataset, labels = make_rectangle(n_samples=n_samples, noise=noise, random_state=42)

# Plot the dataset
plt.figure(figsize=(8, 8))
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.title("Rectangle Shape Dataset")
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
