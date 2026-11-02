
"""
Derivatives, Partial Derivatives and the Chain Rule
Compares exact (formula) derivatives with numerical ones.

"""
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    """Slope of f at x using a tiny step on both sides."""
    return (f(x + h) - f(x - h)) / (2 * h)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

tests = [
    ("x^2 at x = 3",          lambda x: x**2,        lambda x: 2*x,                     3.0),
    ("3x^4+2x^2-5x+7 at x=1", lambda x: 3*x**4 + 2*x**2 - 5*x + 7,
                              lambda x: 12*x**3 + 4*x - 5,                              1.0),
    ("e^x at x = 0",          np.exp,                np.exp,                            0.0),
    ("sigmoid at x = 0",      sigmoid,               lambda x: sigmoid(x)*(1-sigmoid(x)), 0.0),
    ("ReLU at x = 2",         lambda x: max(0, x),   lambda x: 1.0 if x > 0 else 0.0,   2.0),
]
print(f"{'function':26s} {'formula':>10s} {'numerical':>10s}")
for name, f, df, x in tests:
    print(f"{name:26s} {df(x):10.4f} {numerical_derivative(f, x):10.4f}")

# ---- The limit idea: slope of x^2 at x = 1 as h gets smaller ----
print("\nSlope of x^2 at x = 1 with shrinking step h:")
for h in [1, 0.1, 0.01, 0.001]:
    print(f"  h = {h:<6} slope = {((1 + h)**2 - 1) / h:.4f}")

# ---- Partial derivatives: f(x, y) = x^2 y + 3y at (2, 1) ----
f = lambda x, y: x**2 * y + 3 * y
x, y, h = 2.0, 1.0, 1e-5
print("\nf(x, y) = x^2 y + 3y at (2, 1)")
print("  df/dx: formula 2xy =", 2*x*y, "  numerical =",
      round((f(x + h, y) - f(x - h, y)) / (2*h), 4))
print("  df/dy: formula x^2+3 =", x**2 + 3, "  numerical =",
      round((f(x, y + h) - f(x, y - h)) / (2*h), 4))

# ---- Chain rule: y = (3x + 1)^2 -> dy/dx = 2(3x+1) * 3 ----
x = 1.0
u = 3*x + 1
chain = 2*u * 3
print("\nChain rule, y = (3x+1)^2 at x = 1: dy/du * du/dx =", chain,
      "  numerical =", round(numerical_derivative(lambda t: (3*t + 1)**2, x), 4))

# ---- Chain rule inside one neuron: L = (sigmoid(w x + b) - t)^2 ----
xin, w, b, t = 1.0, 0.5, 0.0, 1.0
z = w*xin + b; a = sigmoid(z); L = (a - t)**2
dL_da, da_dz, dz_dw = 2*(a - t), a*(1 - a), xin
print(f"\nNeuron: z = {z}, a = {a:.4f}, L = {L:.4f}")
print(f"dL/dw = {dL_da:.4f} x {da_dz:.4f} x {dz_dw} = {dL_da*da_dz*dz_dw:.4f}")
loss_w = lambda ww: (sigmoid(ww*xin + b) - t)**2
print(f"numerical dL/dw = {numerical_derivative(loss_w, w):.4f}")