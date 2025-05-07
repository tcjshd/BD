import numpy as np

X = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
weights = np.array([[0.2, 0.9], [0.4, 0.7], [0.6, 0.5], [0.8, 0.3]])
learning_rate = 0.5
epochs = 100
for epoch in range(epochs):
  print(f"Iteration {epoch+1}")
  prev_weight = weights.copy()
  for x in X:
    distance = [np.linalg.norm(x - weights[: , j]) for j in range(2)]
    winning = np.argmin(distance)

    weights[: , winning] += learning_rate*(x - weights[: , winning])
  learning_rate *= 0.5
  print(weights)
  if np.array_equal(weights,prev_weight):
    print(f"Convergence met in Iteration {epoch+1}")
    break