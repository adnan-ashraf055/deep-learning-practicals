

"""
Backpropagation Step by Step + Gradient Checking
The worked example from class: the 2-2-1 sigmoid network, target y = 1,
binary cross-entropy, learning rate 0.5.
"""

import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
x, y, lr = np.array([1.0, 0.5]), 1.0, 0.5
params = {
    "W1": np.array([[0.5, -0.4], [0.3, 0.8]]), "b1": np.array([0.1, -0.2]),
    "W2": np.array([[1.2, -0.7]]),             "b2": np.array([0.3]),
}

def forward(p):
    z1 = p["W1"] @ x + p["b1"]; a1 = sigmoid(z1)
    z2 = p["W2"] @ a1 + p["b2"]; a2 = sigmoid(z2)
    loss = -(y * np.log(a2[0]) + (1 - y) * np.log(1 - a2[0]))
    return z1, a1, z2, a2, loss

def backward(p):
    z1, a1, z2, a2, loss = forward(p)
    d2 = a2 - y                                   # 1. output error
    grads = {"W2": np.outer(d2, a1), "b2": d2}    # 3, 4
    d1 = (p["W2"].T @ d2) * a1 * (1 - a1)         # 2. hidden error
    grads["W1"] = np.outer(d1, x); grads["b1"] = d1
    return grads, d2, d1

z1, a1, z2, a2, loss = forward(params)
print("FORWARD:  a1 =", a1.round(4), " y_hat =", a2.round(4), f" loss = {loss:.4f}")

grads, d2, d1 = backward(params)
print("\nBACKWARD")
print("  delta2 = y_hat - y         =", d2.round(4))
print("  dL/dW2 = delta2 * a1       =", grads["W2"].round(4))
print("  W2^T delta2                =", (params["W2"].T @ d2).ravel().round(4))
print("  sigmoid'(z1) = a1(1 - a1)  =", (a1 * (1 - a1)).round(4))
print("  delta1                     =", d1.round(4))
print("  dL/dW1 = delta1 x^T        =\n", grads["W1"].round(4))

# ---- Gradient checking: compare with the numerical slope ----
print("\nGRADIENT CHECK (backprop vs numerical, h = 1e-5)")
h = 1e-5
for name in ["W1", "b1", "W2", "b2"]:
    for idx in np.ndindex(params[name].shape):
        plus  = {k: v.copy() for k, v in params.items()}; plus[name][idx]  += h
        minus = {k: v.copy() for k, v in params.items()}; minus[name][idx] -= h
        numerical = (forward(plus)[4] - forward(minus)[4]) / (2 * h)
        print(f"  {name}{list(idx)}: backprop {grads[name][idx]: .6f}  "
              f"numerical {numerical: .6f}  diff {abs(grads[name][idx] - numerical):.1e}")

# ---- One gradient descent update ----
new = {k: params[k] - lr * grads[k] for k in params}
print("\nUPDATE (lr = 0.5):")
for k in new:
    print(f"  {k}: {params[k].round(4).tolist()} -> {new[k].round(4).tolist()}")
_, _, _, a2_new, loss_new = forward(new)
print(f"\ny_hat: {a2[0]:.4f} -> {a2_new[0]:.4f}   loss: {loss:.4f} -> {loss_new:.4f}")