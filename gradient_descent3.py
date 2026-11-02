
""""
Gradient Descent on a Simple Function
Minimise L(w) = (w - 3)^2 and see the effect of the learning rate.
"""


import numpy as np
import matplotlib.pyplot as plt

loss = lambda w: (w - 3) ** 2
grad = lambda w: 2 * (w - 3)

def gradient_descent(lr, w0=0.0, steps=10):
    w = w0
    path = [w]
    for _ in range(steps):
        w = w - lr * grad(w)          # the update rule
        path.append(w)
    return np.array(path)

print("Step-by-step with learning rate 0.1:")
w = 0.0
for step in range(6):
    g = grad(w)
    new_w = w - 0.1 * g
    print(f"  step {step}: w = {w:.4f}  loss = {loss(w):.4f}  "
          f"gradient = {g:.4f}  new w = {new_w:.4f}")
    w = new_w

print("\nAfter 10 steps:")
for lr in [0.01, 0.1, 1.1]:
    p = gradient_descent(lr)
    print(f"  lr = {lr:<5} w = {p[-1]:9.3f}   loss = {loss(p[-1]):9.3f}")

# ---- Plot the three paths on the loss curve ----
ws = np.linspace(-4, 11, 300)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, lr, title in zip(axes, [0.01, 0.1, 1.1],
                         ["too small", "just right", "too large"]):
    p = gradient_descent(lr, steps=5)
    ax.plot(ws, loss(ws), color="gray")
    ax.plot(p, loss(p), "o-", color="tab:red")
    ax.set_title(f"lr = {lr} ({title})")
    ax.set_xlabel("w"); ax.set_ylabel("loss")
plt.tight_layout()
plt.show()