
"""
Batch vs Stochastic vs Mini-Batch Gradient Descent
One sigmoid neuron (logistic regression) trained with binary cross-entropy.

"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 1000
X = np.vstack([rng.normal(-1, 1, (n // 2, 2)), rng.normal(1, 1, (n // 2, 2))])
y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def cost(w, b):
    p = np.clip(sigmoid(X @ w + b), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train(batch_size, lr=0.1, epochs=10, seed=1):
    r = np.random.default_rng(seed)
    w, b = np.array([-2.0, 2.0]), 0.0         # same (bad) starting point
    history, updates = [cost(w, b)], 0
    for _ in range(epochs):
        idx = r.permutation(n)                  # shuffle every epoch
        for k in range(0, n, batch_size):
            j = idx[k:k + batch_size]
            p = sigmoid(X[j] @ w + b)
            g = p - y[j]                        # dLoss/dz for sigmoid + BCE
            w -= lr * X[j].T @ g / len(j)
            b -= lr * g.mean()
            updates += 1
        history.append(cost(w, b))
    acc = np.mean((sigmoid(X @ w + b) > 0.5) == y)
    return history, updates, acc

plt.figure(figsize=(7, 4))
for name, bs in [("Batch GD (1000)", n), ("Mini-batch (32)", 32), ("SGD (1)", 1)]:
    hist, updates, acc = train(bs)
    print(f"{name:16s}: {updates:5d} updates, final cost {hist[-1]:.4f}, accuracy {acc:.3f}")
    plt.plot(hist, marker="o", label=name)
plt.xlabel("epoch"); plt.ylabel("cost (BCE)")
plt.title("Same data, same learning rate, 10 epochs"); plt.legend(); plt.grid(alpha=0.3)
plt.show()

print("\nIterations per epoch = examples / batch size")
for bs in [60000, 32, 1]:
    print(f"  MNIST, batch size {bs:>5}: {60000 // bs:>6} updates per epoch")