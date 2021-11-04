import numpy as np
import matplotlib.pyplot as plt

# Labels
NASA = 0
ALDI = 1

# Data
data = np.array([[4.9, 10],
                 [4.7, 15],
                 [4.5, 12],
                 [5.8, 25],
                 [4.9, 10],
                 [4, 15],
                 [5, 43],
                 [8.2, 45],
                 [8.7, 50],
                 [6.9, 55],
                 [7.2, 52],
                 [9, 51]])
labels = np.array([NASA, NASA, NASA, NASA, NASA, NASA,
                   ALDI, ALDI, ALDI, ALDI, ALDI, ALDI])

# Means
mean_nasa = np.mean(data[labels == NASA], axis=0)
mean_aldi = np.mean(data[labels == ALDI], axis=0)
print(f"NASA Mean: {mean_nasa}")
print(f"ALDI Mean: {mean_aldi}")

# Cov for data labelled NASA
data_nasa = data[labels == NASA]
N_nasa = data_nasa.shape[0]
cov_nasa = np.zeros((data_nasa.shape[1], data_nasa.shape[1]))
for i in range(N_nasa):
    temp = data_nasa[i] - mean_nasa
    cov_nasa += temp[:, np.newaxis] @ temp[:, np.newaxis].T
cov_nasa /= (N_nasa - 1)

# Cov for data labelled ALDI
data_aldi = data[labels == ALDI]
N_aldi = data_aldi.shape[0]
cov_aldi = np.zeros((data_aldi.shape[1], data_aldi.shape[1]))
for i in range(N_aldi):
    temp = data_aldi[i] - mean_aldi
    cov_aldi += temp[:, np.newaxis] @ temp[:, np.newaxis].T
cov_aldi /= (N_aldi - 1)

print(f"NASA Cov: {cov_nasa}")
print(f"ALDI Cov: {cov_aldi}")

# Within class cov matrix
Sw = 0.5 * (cov_aldi + cov_nasa)
print(f"Sw: {Sw}")

# Find Weight and Bias
W = np.linalg.inv(Sw) @ (mean_aldi - mean_nasa)
b = -0.5 * (W @ (mean_aldi + mean_nasa))
print(f"Weight Vector: {W}, Bias: {b}")

# Predictions on new data
new_x = np.array([[6.0, 25.0]])
pred_labels = np.zeros((new_x.shape[0]))
pred = new_x @ W + b

pred_labels[pred < 0.0] = NASA
pred_labels[pred >= 0.0] = ALDI
print(pred_labels)

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(new_x[:, 0], new_x[:, 1])
plt.show()

