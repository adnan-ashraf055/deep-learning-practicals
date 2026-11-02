
"""
Loss Functions: MSE, MAE, Binary and Categorical Cross-Entropy
All worked examples from class, computed in code.

"""

import numpy as np

def mse(y, y_hat): return np.mean((y - y_hat) ** 2)
def mae(y, y_hat): return np.mean(np.abs(y - y_hat))

def binary_cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)                 # avoid log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def categorical_cross_entropy(Y_onehot, P, eps=1e-12):
    P = np.clip(P, eps, 1)
    return -np.mean(np.sum(Y_onehot * np.log(P), axis=1))

# ---- Regression: house prices in lakh ----
y     = np.array([50, 40, 70])
y_hat = np.array([45, 42, 60])
print("House prices: errors =", y - y_hat)
print(f"  MSE  = {mse(y, y_hat):.2f}")
print(f"  RMSE = {np.sqrt(mse(y, y_hat)):.2f} lakh")
print(f"  MAE  = {mae(y, y_hat):.2f} lakh")

# ---- Binary cross-entropy: spam filter ----
y = np.array([1, 1, 1, 0])
p = np.array([0.9, 0.6, 0.1, 0.2])
print("\nSpam filter, loss per email:")
for yi, pi in zip(y, p):
    print(f"  y = {yi}, p = {pi}: loss = {binary_cross_entropy(np.array([yi]), np.array([pi])):.4f}")
print(f"  average (cost) = {binary_cross_entropy(y, p):.4f}")

# ---- Why cross-entropy, not MSE, for classification ----
print("\nTrue label 1:   p     MSE     BCE")
for pi in [0.9, 0.5, 0.1, 0.01]:
    print(f"            {pi:5}  {(1 - pi)**2:6.3f}  {-np.log(pi):6.3f}")

# ---- Categorical cross-entropy: cat / dog / bird ----
P = np.array([[0.836, 0.113, 0.051],
              [0.836, 0.113, 0.051]])
Y = np.array([[1, 0, 0],      # true class: cat
              [0, 0, 1]])     # true class: bird
print("\nCategorical cross-entropy:")
for i, name in enumerate(["cat", "bird"]):
    print(f"  true = {name:4s}: loss = {categorical_cross_entropy(Y[i:i+1], P[i:i+1]):.3f}")

# ---- Sparse labels (integers) give the same result ----
labels = np.array([0, 2])
sparse = -np.mean(np.log(P[np.arange(2), labels]))
print(f"  average with integer labels {labels}: {sparse:.3f} "
      f"= one-hot version {categorical_cross_entropy(Y, P):.3f}")