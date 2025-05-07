import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize pixel values (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encode labels (e.g., 3 → [0,0,0,1,0,0,0,0,0,0])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. RNN Model
model = Sequential([
    SimpleRNN(64, input_shape=(28, 28), activation='tanh'),
    Dense(10, activation='softmax')  # 10 output classes (0-9)
])

# 5. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 7. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# 8. Predict a sample
sample = np.expand_dims(x_test[0], axis=0)  # Shape: (1, 28, 28)
prediction = model.predict(sample)
predicted_digit = np.argmax(prediction)

plt.imshow(x_test[0], cmap='gray')
plt.title(f"True Label: {np.argmax(y_test[0])}, Predicted: {predicted_digit}")
plt.axis('off')
plt.show()