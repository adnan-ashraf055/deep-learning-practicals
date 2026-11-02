
"""
An MLP Trained with Backpropagation (from scratch): XOR
A 2-4-1 network with sigmoid units learns XOR by itself.

"""

import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, n_in, n_hidden, n_out, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1, (n_in, n_hidden)); self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1, (n_hidden, n_out)); self.b2 = np.zeros(n_out)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.a1 = self.sigmoid(X @ self.W1 + self.b1)      # hidden layer
        self.p = self.sigmoid(self.a1 @ self.W2 + self.b2) # output layer
        return self.p

    def backward(self, X, y, lr):
        n = len(X)
        d2 = (self.p - y) / n                                # output error
        d1 = (d2 @ self.W2.T) * self.a1 * (1 - self.a1)      # hidden error
        self.W2 -= lr * self.a1.T @ d2; self.b2 -= lr * d2.sum(axis=0)
        self.W1 -= lr * X.T @ d1;       self.b1 -= lr * d1.sum(axis=0)

    @staticmethod
    def bce(y, p):
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y, epochs=5000, lr=1.0):
        losses = []
        for epoch in range(1, epochs + 1):
            p = self.forward(X)
            losses.append(self.bce(y, p))
            self.backward(X, y, lr)
            if epoch in (1, 500, 1000, 2000, 5000):
                print(f"epoch {epoch:4d}: loss {losses[-1]:.4f}")
        return losses

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

net = MLP(2, 4, 1)
losses = net.fit(X, y)
print("\nPredictions:", net.forward(X).ravel().round(3), " targets:", y.ravel())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(losses); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
ax[0].set_title("Training loss"); ax[0].grid(alpha=0.3)
g = np.linspace(-0.5, 1.5, 200); G1, G2 = np.meshgrid(g, g)
P = net.forward(np.c_[G1.ravel(), G2.ravel()]).reshape(G1.shape)
ax[1].contourf(G1, G2, P, levels=20, cmap="coolwarm", alpha=0.8)
ax[1].scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", s=200, edgecolors="k")
ax[1].set_title("Learned decision regions (not a straight line!)")
plt.tight_layout()
plt.show()