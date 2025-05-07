import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Specify the path to your image file
# Replace this with the path to your image
image_path = input("Enter the path to your image file: ")

# Check if file exists
if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
    # Read the image
    image = cv2.imread(image_path)
    
    # Display original image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(1, 2, 2)
    plt.title("Grayscale Image")
    plt.imshow(gray, cmap='gray')
    plt.show()
    
    # Convolution function
    def convolution(image, filter, padding=0, stride=1):
        img_h, img_w = image.shape
        filt_h, filt_w = filter.shape
        if padding > 0:
            image = np.pad(image, ((padding, padding), (padding, padding)), 
                          mode='constant', constant_values=0)
        out_h = (image.shape[0] - filt_h) // stride + 1
        out_w = (image.shape[1] - filt_w) // stride + 1
        result = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                patch = image[i*stride:i*stride+filt_h, j*stride:j*stride+filt_w]
                value = np.sum(patch * filter)
                result[i, j] = max(0, value)
        return result
    
    # Pooling function
    def pooling(image, pool_size=2, stride=1):
        img_h, img_w = image.shape
        out_h = ((img_h - pool_size)) // stride + 1
        out_w = ((img_w - pool_size)) // stride + 1
        result = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                patch = image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                result[i, j] = np.max(patch)
        return result
    
    # Apply convolution (sharpening filter)
    conv_img = convolution(gray, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    
    # Apply pooling
    pool_img = pooling(conv_img)
    
    # Display processed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("After Convolution")
    plt.imshow(conv_img, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("After Pooling")
    plt.imshow(pool_img, cmap='gray')
    plt.show()
    
    # Flatten and apply weights
    flattened = pool_img.flatten()
    weight = np.random.rand(flattened.shape[0])
    net = np.dot(flattened, weight)
    out = 1/(1+np.exp(-net))
    print(f"Output after sigmoid activation: {out}")
    
