"""
Optimizers from Scratch: SGD, Momentum, RMSProp, Adam
All four minimise the same "ravine" loss L = 0.5 * (w1^2 + 50 * w2^2).

"""

import numpy as np
import matplotlib.pyplot as plt

A = np.array([1.0, 50.0])
loss = lambda w: 0.5 * np.sum(A * w ** 2)
grad = lambda w: A * w
start, steps = np.array([-4.0, 0.8]), 50

def sgd(lr=0.038):
    w = start.copy(); path = [w.copy()]
    for _ in range(steps):
        w -= lr * grad(w); path.append(w.copy())
    return np.array(path)

def momentum(lr=0.012, beta=0.9):
    w, v = start.copy(), np.zeros(2); path = [w.copy()]
    for _ in range(steps):
        v = beta * v + grad(w)
        w -= lr * v; path.append(w.copy())
    return np.array(path)

def rmsprop(lr=0.2, rho=0.9, eps=1e-8):
    w, s = start.copy(), np.zeros(2); path = [w.copy()]
    for _ in range(steps):
        g = grad(w)
        s = rho * s + (1 - rho) * g ** 2
        w -= lr * g / (np.sqrt(s) + eps); path.append(w.copy())
    return np.array(path)

def adam(lr=0.2, b1=0.9, b2=0.999, eps=1e-8):
    w, m, v = start.copy(), np.zeros(2), np.zeros(2); path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps); path.append(w.copy())
    return np.array(path)

runs = {"SGD": sgd(), "Momentum": momentum(), "RMSProp": rmsprop(), "Adam": adam()}
print(f"Starting loss: {loss(start):.2f}")
for name, path in runs.items():
    print(f"{name:9s} loss after 10 steps {loss(path[10]):8.4f}   after 50 steps {loss(path[-1]):.4f}")

# Adam's first step is about lr whatever the size of the gradient
for g in [4.0, 0.04]:
    m, v = 0.1 * g, 0.001 * g * g
    print(f"Adam first step with gradient {g}: lr x {(m / 0.1) / np.sqrt(v / 0.001):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
t = np.linspace(-4.5, 1.5, 300); u = np.linspace(-1, 1, 300); T, U = np.meshgrid(t, u)
ax[0].contour(T, U, 0.5 * (T ** 2 + 50 * U ** 2), levels=15, cmap="Greys")
for name, path in runs.items():
    ax[0].plot(path[:, 0], path[:, 1], ".-", label=name)
ax[0].set_title("Paths on the ravine"); ax[0].legend()
for name, path in runs.items():
    ax[1].semilogy([loss(p) + 1e-6 for p in path], label=name)
ax[1].set_xlabel("step"); ax[1].set_ylabel("loss (log scale)"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()