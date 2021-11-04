import numpy as np

data = np.array([[2, 4],
                 [3, 3],
                 [-4, -2],
                 [-2, 6]])
labels = np.array([1, 1, 0, 0])
data = np.hstack((np.ones((data.shape[0], 1)), data))

learning_rate = 0.5
W = np.array([0.0, 0.0, 0.0])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Run one step of Gradient Descent
for iteration in range(0, 1):
    grad = -np.sum(data * labels[:, np.newaxis] - sigmoid(data) * data, axis=0)
    W += learning_rate * grad
print(W)

def predict(X):
    pred = sigmoid(X @ W)
    print("Prediction: ", pred)

# Prediction for a new sample
new_X = np.array([1, -1, 1])
predict(new_X)

