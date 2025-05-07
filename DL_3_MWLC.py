# %pip install matplotlib
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (_, _) = mnist.load_data()

image = X_train[0] / 255.0
label = y_train[0]
print(f"True Label: {label}")

plt.imshow(image, cmap='gray')
plt.title(f"Digit: {label}")
plt.show()

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

def convolve2d(img, kernel):
    kernel_size = kernel.shape[0]
    output_dim = img.shape[0] - kernel_size + 1
    result = np.zeros((output_dim, output_dim))
    for i in range(output_dim):
        for j in range(output_dim):
            region = img[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(region * kernel)
    return result

def max_pooling(feature_map, size=2, stride=2):
    h, w = feature_map.shape
    pooled_h = (h - size) // stride + 1
    pooled_w = (w - size) // stride + 1
    pooled = np.zeros((pooled_h, pooled_w))

    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            region = feature_map[i:i+size, j:j+size]
            pooled[i//stride, j//stride] = np.max(region)
    return pooled


def fully_connected(flattened_input, weights, bias):
    return np.dot(flattened_input, weights) + bias

conv_output = convolve2d(image, kernel)
pooled_output = max_pooling(conv_output)
flattened = pooled_output.flatten()

weights = np.random.randn(flattened.size, 10)
bias = np.random.randn(10)

fc_output = fully_connected(flattened, weights, bias)
predicted = np.argmax(fc_output)
# print(predicted)
print("Predicted Digit:", predicted) 
