import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# XOR inputs and expected outputs
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Learning parameters
epochs = 10000
lr = 0.5
display_step = 1000

# Initialize weights and biases randomly
np.random.seed(42)
# Hidden layer: 2 inputs -> 2 neurons
w_hidden = np.random.rand(2, 2) 
b_hidden = np.random.rand(1, 2)
# Output layer: 2 hidden neurons -> 1 output
w_output = np.random.rand(2, 1)
b_output = np.random.rand(1, 1)

print("Initial weights and biases:")
print(f"Hidden weights:\n{w_hidden}")
print(f"Hidden bias:\n{b_hidden}")
print(f"Output weights:\n{w_output}")
print(f"Output bias:\n{b_output}")
print("\nTraining the XOR neural network:")

# Training
for i in range(epochs):
    # Forward Pass
    # Hidden layer
    hidden_layer_input = np.dot(x, w_hidden) + b_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Output layer
    output_layer_input = np.dot(hidden_layer_output, w_output) + b_output
    predicted_output = sigmoid(output_layer_input)
    
    # Calculate error
    error = y - predicted_output
    
    if (i % display_step == 0):
        loss = np.mean(np.square(error))
        print(f"Epoch {i}: Loss = {loss:.6f}")
    
    # Backward Pass
    # Output layer
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Hidden layer
    error_hidden_layer = d_predicted_output.dot(w_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    # Output layer
    w_output += hidden_layer_output.T.dot(d_predicted_output) * lr
    b_output += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    
    # Hidden layer
    w_hidden += x.T.dot(d_hidden_layer) * lr
    b_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("\nTraining Complete!")
print(f"Final weights and biases:")
print(f"Hidden weights:\n{w_hidden}")
print(f"Hidden bias:\n{b_hidden}")
print(f"Output weights:\n{w_output}")
print(f"Output bias:\n{b_output}")

# Testing
print("\nPredictions after training:")
hidden_layer_output = sigmoid(np.dot(x, w_hidden) + b_hidden)
predicted_output = sigmoid(np.dot(hidden_layer_output, w_output) + b_output)

for i in range(len(x)):
    print(f"Input: {x[i]} -> Predicted: {predicted_output[i][0]:.4f} | Expected: {y[i][0]}")

# Visualization of decision making
def make_prediction(input_data):
    hidden = sigmoid(np.dot(input_data, w_hidden) + b_hidden)
    output = sigmoid(np.dot(hidden, w_output) + b_output)
    return output[0][0]

print("\nDetailed forward pass for input [0, 1]:")
test_input = np.array([[0, 1]])
hidden_test = sigmoid(np.dot(test_input, w_hidden) + b_hidden)
output_test = sigmoid(np.dot(hidden_test, w_output) + b_output)

print(f"Hidden layer activations: {hidden_test[0]}")
print(f"Final output: {output_test[0][0]:.4f}")