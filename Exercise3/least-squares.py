import numpy as np

X = np.array([[-0.8, 2.8],
              [0.3, -2.2],
              [1.5, 1.1]])
y = np.array([-8.5, 12.8, 3.8])

W = np.linalg.inv(X.T @ X) @ X.T @ y
print("W: ", W)

X_updated = np.hstack((np.ones((X.shape[0], 1)), X))

W_with_bias = np.linalg.inv(X_updated.T @ X_updated) @ X_updated.T @ y
print("W with bias: ", W_with_bias)


def true_model(x1, x2):
    return 5 + 2 * x1 - 4 * x2

X_test = np.array([[-2, 2],
                   [-4, 15]])
y_test = true_model(X_test[:, 0], X_test[:, 1])
print("X_test: ", X_test, "y_test: ", y_test)

print("Prediction using h1: ", X_test @ W)
print("Prediction using h2: ", np.hstack((np.ones((X_test.shape[0], 1)), X_test)) @ W_with_bias)

print(f"Prediction W {X @ W}")
print(f"Prediction- bias {X_updated @ W_with_bias}")