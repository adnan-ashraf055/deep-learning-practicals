

"""
Solving XOR with a Two-Layer Network (hand-set weights)
Hidden layer: h1 = OR, h2 = NAND.  Output: AND(h1, h2) = XOR.

"""

import numpy as np

step = lambda z: (z > 0).astype(int)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W1 = np.array([[ 1,  1],      # h1 = OR
               [-1, -1]])     # h2 = NAND
b1 = np.array([-0.5, 1.5])
W2 = np.array([1, 1])         # output = AND(h1, h2)
b2 = -1.5

H = step(X @ W1.T + b1)       # hidden layer for all 4 inputs at once
y = step(H @ W2 + b2)

print(" x1 x2 | h1(OR) h2(NAND) | y = XOR")
for xi, hi, yi in zip(X, H, y):
    print(f"  {xi[0]}  {xi[1]} |   {hi[0]}      {hi[1]}     |   {yi}")

print("\nThe hidden layer moves the points into a new space (h1, h2):")
for xi, hi, yi in zip(X, H, y):
    print(f"  input {tuple(int(v) for v in xi)} -> hidden {tuple(int(v) for v in hi)}   class {yi}")
print("In hidden space the two class-1 points both land on (1, 1),")
print("so a single straight line h1 + h2 = 1.5 now separates the classes.")