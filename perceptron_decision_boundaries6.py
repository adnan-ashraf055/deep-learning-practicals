
"""
Visualising a Perceptron's Decision Boundary
The perceptron's boundary is the straight line w1*x1 + w2*x2 + b = 0.

"""

import numpy as np
import matplotlib.pyplot as plt

def train(X, y, lr=0.5, epochs=100):
    w, b = np.zeros(2), 0.0
    for _ in range(epochs):
        errors = 0
        for xi, t in zip(X, y):
            e = t - (1 if w @ xi + b > 0 else 0)
            w += lr * e * xi; b += lr * e; errors += abs(e)
        if errors == 0:
            break
    return w, b

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
gates = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
xs = np.linspace(-0.5, 1.5, 100)
for ax, (name, y) in zip(axes, gates.items()):
    y = np.array(y)
    w, b = train(X, y)
    acc = np.mean([(1 if w @ x + b > 0 else 0) == t for x, t in zip(X, y)])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=200, edgecolors="k")
    if w[1] != 0:
        ax.plot(xs, -(w[0] * xs + b) / w[1], "g-", lw=2)
    elif w[0] != 0:
        ax.axvline(-b / w[0], color="g", lw=2)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_title(f"{name}: accuracy {acc:.0%}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    print(f"{name}: w = {w}, b = {b}, accuracy = {acc:.0%}")
plt.tight_layout()
plt.show()