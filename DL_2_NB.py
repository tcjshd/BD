import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# Inputs and expected outputs
x = np.array([0.1, 0.5])
output = np.array([0.05, 0.95])
lr = 0.6  # learning rate

# Initial weights and biases
w1, w2 = 0.1, 0.2
w3, w4 = 0.3, 0.4
w5, w6 = 0.5, 0.6
w7, w8 = 0.7, 0.8

b1, b2 = 0.25, 0.25
b3, b4 = 0.35, 0.35

### --------- FORWARD PASS ---------
# Hidden Layer
net_h1 = x[0]*w1 + x[1]*w3 + b1
out_h1 = sigmoid(net_h1)

net_h2 = x[0]*w2 + x[1]*w4 + b2
out_h2 = sigmoid(net_h2)

# Output Layer
net_o1 = out_h1*w5 + out_h2*w6 + b3
out_o1 = sigmoid(net_o1)

net_o2 = out_h1*w7 + out_h2*w8 + b4
out_o2 = sigmoid(net_o2)

# Total Error
E_total = 0.5 * ((output[0] - out_o1)**2 + (output[1] - out_o2)**2)

print("=== Forward Pass ===")
print(f"Net h1: {net_h1}, Out h1: {out_h1}")
print(f"Net h2: {net_h2}, Out h2: {out_h2}")
print(f"Net o1: {net_o1}, Out o1: {out_o1}")
print(f"Net o2: {net_o2}, Out o2: {out_o2}")
print(f"Total Error: {E_total:.6f}")

### --------- BACKWARD PASS ---------
# Derivatives of error w.r.t. output
dE_o1 = out_o1 - output[0]
dE_o2 = out_o2 - output[1]

# Derivatives of output w.r.t. net input (sigmoid derivative)
d_out_o1 = sigmoid_derivative(out_o1)
d_out_o2 = sigmoid_derivative(out_o2)

# Gradients for output weights
dE_w5 = dE_o1 * d_out_o1 * out_h1
dE_w6 = dE_o1 * d_out_o1 * out_h2
dE_w7 = dE_o2 * d_out_o2 * out_h1
dE_w8 = dE_o2 * d_out_o2 * out_h2

# Update output weights
w5 -= lr * dE_w5
w6 -= lr * dE_w6
w7 -= lr * dE_w7
w8 -= lr * dE_w8

# Gradients for hidden neurons (backpropagated from both output neurons)
d_hidden1 = ((dE_o1 * d_out_o1 * w5) + (dE_o2 * d_out_o2 * w7)) * sigmoid_derivative(out_h1)
d_hidden2 = ((dE_o1 * d_out_o1 * w6) + (dE_o2 * d_out_o2 * w8)) * sigmoid_derivative(out_h2)

# Gradients for hidden weights
dE_w1 = d_hidden1 * x[0]
dE_w2 = d_hidden2 * x[0]
dE_w3 = d_hidden1 * x[1]
dE_w4 = d_hidden2 * x[1]

# Update hidden weights
w1 -= lr * dE_w1
w2 -= lr * dE_w2
w3 -= lr * dE_w3
w4 -= lr * dE_w4

print("\n=== Backward Pass ===")
print(f"Updated Output Weights: w5={w5:.4f}, w6={w6:.4f}, w7={w7:.4f}, w8={w8:.4f}")
print(f"Updated Hidden Weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, w4={w4:.4f}")
