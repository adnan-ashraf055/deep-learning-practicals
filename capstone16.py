
"""
(Capstone): A Complete MLP Classifier from Scratch in NumPy
Recognises handwritten digits (8 x 8 images, 10 classes) using everything
from Units 1-2: ReLU hidden layer, softmax output, categorical cross-entropy,
mini-batch training, backpropagation and the Adam optimizer.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---- 1. Data ----
digits = load_digits()                         # 1,797 images, 8 x 8 pixels
X = digits.data / 16.0                         # scale pixels to 0..1
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
Y_train = np.eye(10)[y_train]                  # one-hot labels
print("train", X_train.shape, "test", X_test.shape)

# ---- 2. Network: 64 -> 64 (ReLU) -> 10 (softmax) ----
rng = np.random.default_rng(0)
params = {
    "W1": rng.normal(0, np.sqrt(2 / 64), (64, 64)), "b1": np.zeros(64),
    "W2": rng.normal(0, np.sqrt(2 / 64), (64, 10)), "b2": np.zeros(10),
}
print("parameters:", sum(p.size for p in params.values()))

def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(0, z1)                               # ReLU
    z2 = a1 @ p["W2"] + p["b2"]
    e = np.exp(z2 - z2.max(axis=1, keepdims=True))
    return z1, a1, e / e.sum(axis=1, keepdims=True)      # softmax

def backward(X, Y, p, cache):
    z1, a1, P = cache
    n = len(X)
    d2 = (P - Y) / n                                     # softmax + CCE
    d1 = (d2 @ p["W2"].T) * (z1 > 0)                     # ReLU slope
    return {"W2": a1.T @ d2, "b2": d2.sum(0), "W1": X.T @ d1, "b1": d1.sum(0)}

# ---- 3. Adam optimizer ----
lr, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}
t = 0

def accuracy(X, y):
    return np.mean(forward(X, params)[2].argmax(axis=1) == y)

# ---- 4. Training loop: mini-batches of 32 ----
history = {"loss": [], "train_acc": [], "test_acc": []}
for epoch in range(1, 31):
    idx = rng.permutation(len(X_train))
    for k in range(0, len(idx), 32):
        j = idx[k:k + 32]
        cache = forward(X_train[j], params)
        grads = backward(X_train[j], Y_train[j], params, cache)
        t += 1
        for name in params:
            m[name] = b1 * m[name] + (1 - b1) * grads[name]
            v[name] = b2 * v[name] + (1 - b2) * grads[name] ** 2
            m_hat = m[name] / (1 - b1 ** t); v_hat = v[name] / (1 - b2 ** t)
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    P = forward(X_train, params)[2]
    loss = -np.mean(np.log(P[np.arange(len(y_train)), y_train] + 1e-12))
    history["loss"].append(loss)
    history["train_acc"].append(accuracy(X_train, y_train))
    history["test_acc"].append(accuracy(X_test, y_test))
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:2d}: loss {loss:.4f}  train acc {history['train_acc'][-1]:.3f}"
              f"  test acc {history['test_acc'][-1]:.3f}")

# ---- 5. Results ----
pred = forward(X_test, params)[2].argmax(axis=1)
print(f"\nFinal test accuracy: {np.mean(pred == y_test):.3f}")
cm = np.zeros((10, 10), dtype=int)
for t_, p_ in zip(y_test, pred):
    cm[t_, p_] += 1
print("Confusion matrix (rows = true digit, columns = predicted):")
print(cm)

fig = plt.figure(figsize=(13, 4))
ax1 = fig.add_subplot(1, 3, 1); ax1.plot(history["loss"]); ax1.set_title("Training loss")
ax1.set_xlabel("epoch"); ax1.grid(alpha=0.3)
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(history["train_acc"], label="train"); ax2.plot(history["test_acc"], label="test")
ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(alpha=0.3)
for i in range(8):
    a = fig.add_subplot(2, 12, 9 + i if i < 4 else 17 + i)
    a.imshow(X_test[i].reshape(8, 8), cmap="gray_r"); a.axis("off")
    a.set_title(f"pred {pred[i]}", fontsize=9, color="green" if pred[i] == y_test[i] else "red")
plt.tight_layout()
plt.show()