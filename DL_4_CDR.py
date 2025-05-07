# %pip install matplotlib
# %pip install numpy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SimpleRNN, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
# 1. Image Preprocessing
img_size = (150, 150)  # Resizing images
batch_size = 32

# ImageDataGenerator for data augmentation and validation split
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize image pixel values to the range [0, 1]
    validation_split=0.2  # Split 20% for validation
)

# Train generator using the 'PetImages' directory
train_generator = datagen.flow_from_directory(
    'PetImages',  # Path to the main directory
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',  # 2 classes: cat and dog
    subset='training'  # Specify that this generator is for training data
)

# Validation generator using the same 'PetImages' directory
val_generator = datagen.flow_from_directory(
    'PetImages',  # Path to the main directory
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',  # 2 classes: cat and dog
    subset='validation'  # Specify that this generator is for validation data
)

# 2. Build the CNN-RNN Hybrid Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Reshape((1, -1)),  # Reshape the output into (batch_size, timesteps, features) for RNN
    SimpleRNN(64, activation='tanh'),
    Dense(1, activation='sigmoid')  # Binary output (cat or dog)
])

# 3. Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# 5. Evaluate the Model
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc:.4f}")

# 6. Predict a Sample
sample = next(val_generator)  # Get the next batch from the validation generator
image, label = sample[0][0], sample[1][0]  # Get the first image and its label

# Predict on the sample image
prediction = model.predict(np.expand_dims(image, axis=0))  # Predict on the sample image
predicted_label = 1 if prediction > 0.5 else 0  # Binary classification: 1 = dog, 0 = cat

# Visualize the image and prediction
plt.imshow(image)
plt.title(f"True Label: {'Dog' if label == 1 else 'Cat'}, Predicted: {'Dog' if predicted_label == 1 else 'Cat'}")
plt.axis('off')
plt.show()
