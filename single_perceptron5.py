
"""
Single Perceptron with the Perceptron Learning Rule
Trains AND, OR and NAND gates, printing every update, and shows that
XOR can never be learned by a single perceptron.

"""

import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.5, w=None, b=0.0):
        self.w = np.zeros(n_inputs) if w is None else np.array(w, dtype=float)
        self.b = float(b)
        self.lr = lr

    def predict(self, x):
        return 1 if np.dot(self.w, x) + self.b > 0 else 0

    def fit(self, X, y, max_epochs=20, verbose=False):
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                z = np.dot(self.w, xi) + self.b
                output = 1 if z > 0 else 0
                error = target - output
                if verbose:
                    print(f"  epoch {epoch} x={xi} t={target} z={z:5.2f} "
                          f"y={output} e={error:+d}", end="")
                # perceptron learning rule
                self.w += self.lr * error * xi
                self.b += self.lr * error
                errors += abs(error)
                if verbose:
                    print(f" -> w={self.w}, b={self.b:.2f}")
            if errors == 0:
                return epoch          # converged
        return None                   # did not converge

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

print("AND gate (w = [1, 1], b = 0, lr = 0.5), every update shown:")
p = Perceptron(2, lr=0.5, w=[1, 1], b=0)
ep = p.fit(X, np.array([0, 0, 0, 1]), verbose=True)
print(f"Converged in {ep} epochs. Final w = {p.w}, b = {p.b}")
print("Predictions:", [p.predict(x) for x in X])

for name, y, w0, b0 in [("OR", [0, 1, 1, 1], [0, 0], -0.5),
                        ("NAND", [1, 1, 1, 0], [0, 0], 0)]:
    p = Perceptron(2, lr=0.5 if name == "OR" else 1.0, w=w0, b=b0)
    ep = p.fit(X, np.array(y))
    print(f"\n{name}: converged in {ep} epochs, w = {p.w}, b = {p.b}, "
          f"predictions = {[p.predict(x) for x in X]}")

p = Perceptron(2, lr=0.5)
ep = p.fit(X, np.array([0, 1, 1, 0]), max_epochs=100)
print(f"\nXOR: converged? {ep is not None}. After 100 epochs predictions = "
      f"{[p.predict(x) for x in X]} (target [0, 1, 1, 0])")
print("A single perceptron draws one straight line, and XOR is not linearly separable.")