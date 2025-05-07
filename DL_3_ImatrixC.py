
//Without Library 
import numpy as np

image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 2, 2, 0],
    [2, 1, 0, 1, 2],
    [0, 1, 3, 1, 0]
])

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

def max_pooling(feature_map, size=2, stride=1):
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
print("After Convolution:\n", conv_output)

pooled_output = max_pooling(conv_output)
print("After Pooling:\n", pooled_output)

flattened = pooled_output.flatten()
print("After Flattening:\n", flattened)

weights = np.random.randn(flattened.size, 2)
bias = np.random.randn(2)

fc_output = fully_connected(flattened, weights, bias)
print("Final Output (FC Layer):\n", fc_output)


//With Library 
import numpy as np
from keras.models import Sequential                          
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 2, 2, 0],
    [2, 1, 0, 1, 2],
    [0, 1, 3, 1, 0]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

model = Sequential([
   Conv2D(filters=1, kernel_size=(3, 3), padding='valid', activation='linear', input_shape=(5, 5, 1), use_bias=False),
   MaxPooling2D(pool_size=(2, 2)),
   Flatten(),
   Dense(units=1, activation='sigmoid')
])
model.layers[0].set_weights([kernel.reshape(3,3,1,1)])
model.compile(optimizer='adam', loss='mean_squared_error')

output = model.predict(image.reshape(1,5,5,1))
print(f"The output value is {output}")
