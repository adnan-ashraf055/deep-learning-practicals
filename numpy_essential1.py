"""
NumPy Essentials for Deep Learning
Topics: scalars, vectors, matrices, tensors, shapes, dot product,
 matrix multiplication, transpose, broadcasting, a layer z = Wx + b
"""

import numpy as np

# ---- 1. Scalars, vectors, matrices, tensors ----
scalar = np.array(7)
vector = np.array([3, 1, 4, 1])
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
tensor = np.zeros((2, 3, 4))          # e.g. 2 images, 3 rows, 4 columns

for name, t in [("scalar", scalar), ("vector", vector),
                ("matrix", matrix), ("tensor", tensor)]:
    print(f"{name:7s} ndim = {t.ndim}, shape = {t.shape}")

# ---- 2. Element-wise operations ----
a = np.array([2, 1, 3])
b = np.array([1, 4, -1])
print("\na + b        =", a + b)
print("3 * b        =", 3 * b)
print("a * b        =", a * b, "  (element-wise, NOT matrix multiplication)")

# ---- 3. Dot product ----
a = np.array([2, 3, -1])
b = np.array([4, 0, 5])
print("\nDot product a . b =", np.dot(a, b), "  (2*4 + 3*0 + (-1)*5 = 3)")

# ---- 4. Matrix multiplication and the shape rule ----
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("\nA @ B =\n", A @ B)
print("B @ A =\n", B @ A, "\n-> order matters: AB != BA")

M1 = np.ones((3, 4)); M2 = np.ones((4, 2))
print("\n(3 x 4) @ (4 x 2) gives shape", (M1 @ M2).shape)
try:
    np.ones((2, 3)) @ np.ones((2, 3))
except ValueError as e:
    print("(2 x 3) @ (2 x 3) fails:", str(e)[:60], "...")

# ---- 5. Transpose ----
print("\nmatrix shape", matrix.shape, "-> matrix.T shape", matrix.T.shape)
print(matrix.T)

# ---- 6. One neural network layer: z = W x + b ----
x = np.array([1, 2, 3])                     # 3 inputs
W = np.array([[0.2, -0.5,  0.1],            # 2 neurons x 3 inputs
              [0.4,  0.3, -0.2]])
b = np.array([0.1, -0.3])
z = W @ x + b
print("\nLayer: z = W x + b =", z.round(4))
print("ReLU(z)            =", np.maximum(0, z).round(4))

# ---- 7. Broadcasting: a batch of 4 examples through the same layer ----
X = np.array([[1, 2, 3],
              [0, 1, 0],
              [2, 2, 2],
              [1, 0, 1]])                    # (4 examples x 3 features)
Z = X @ W.T + b                             # b is added to every row
print("\nBatch output shape:", Z.shape)
print(Z.round(4))