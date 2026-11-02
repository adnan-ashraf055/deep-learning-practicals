
"""
Softmax: Turning Scores into Probabilities
Includes the numerically stable version used in real code.

"""

import numpy as np

def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax(z):
    e = np.exp(z - np.max(z))     # subtract the max: same answer, no overflow
    return e / e.sum()

z = np.array([3.0, 1.0, 0.2])     # scores for cat, dog, bird
e = np.exp(z)
print("Scores z      :", z)
print("Step 1  e^z   :", e.round(3))
print("Step 2  sum   :", e.sum().round(3))
print("Step 3  probs :", softmax(z).round(3), f" sum = {softmax(z).sum():.4f}")
print("Prediction    :", ["cat", "dog", "bird"][np.argmax(softmax(z))])

print("\nHomework check, z = [2.0, 1.0, 0.1]:", softmax(np.array([2.0, 1.0, 0.1])).round(3))

big = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("\nLarge scores, naive softmax :", softmax_naive(big))
print("Large scores, stable softmax:", softmax(big).round(4))

# A batch: softmax applied row by row (one row per example)
Z = np.array([[3.0, 1.0, 0.2],
              [0.5, 2.5, 0.5]])
P = np.exp(Z - Z.max(axis=1, keepdims=True))
P = P / P.sum(axis=1, keepdims=True)
print("\nBatch probabilities:\n", P.round(3), "\nPredicted classes:", P.argmax(axis=1))