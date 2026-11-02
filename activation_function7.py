"""
Activation Functions and Their Derivatives
Step, Sigmoid, Tanh, ReLU, Leaky ReLU: values, derivatives and plots.

"""


import numpy as np
import matplotlib.pyplot as plt

def step(x):        return np.where(x > 0, 1.0, 0.0)
def sigmoid(x):     return 1 / (1 + np.exp(-x))
def tanh(x):        return np.tanh(x)
def relu(x):        return np.maximum(0, x)
def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)

def d_sigmoid(x):   return sigmoid(x) * (1 - sigmoid(x))
def d_tanh(x):      return 1 - np.tanh(x) ** 2
def d_relu(x):      return np.where(x > 0, 1.0, 0.0)
def d_leaky(x, a=0.01): return np.where(x > 0, 1.0, a)

# ---- Table of values (the class exercise) ----
xs = np.array([-3, -1, 0, 0.5, 3])
print(f"{'x':>5} {'step':>6} {'sigmoid':>8} {'tanh':>8} {'ReLU':>6} {'Leaky':>7}")
for x in xs:
    print(f"{x:5.1f} {step(x):6.0f} {sigmoid(x):8.4f} {tanh(x):8.4f} "
          f"{relu(x):6.2f} {leaky_relu(x):7.3f}")

# ---- Derivatives: why sigmoid causes vanishing gradients ----
print("\nSlope of sigmoid:")
for x in [-10, -4, 0, 4, 10]:
    print(f"  x = {x:4d}: sigmoid'(x) = {d_sigmoid(x):.6f}")
print("Maximum slope of sigmoid = 0.25, so 5 sigmoid layers can shrink a")
print("gradient by up to 0.25**5 =", 0.25 ** 5)

# ---- Plots: functions (top) and derivatives (bottom) ----
x = np.linspace(-5, 5, 400)
funcs = [("Step", step, lambda v: np.zeros_like(v)), ("Sigmoid", sigmoid, d_sigmoid),
         ("Tanh", tanh, d_tanh), ("ReLU", relu, d_relu),
         ("Leaky ReLU (a=0.1)", lambda v: leaky_relu(v, 0.1), lambda v: d_leaky(v, 0.1))]
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
for i, (name, f, df) in enumerate(funcs):
    axes[0, i].plot(x, f(x), lw=2.5); axes[0, i].set_title(name)
    axes[1, i].plot(x, df(x), lw=2.5, color="tab:red")
    axes[1, i].set_title(f"slope of {name.split(' (')[0]}")
    for ax in axes[:, i]:
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
        ax.grid(alpha=0.3)
axes[1, 0].set_ylim(-0.1, 1.1)
axes[1, 0].set_title("slope of Step: 0 (undefined at 0)")
plt.tight_layout()
plt.show()