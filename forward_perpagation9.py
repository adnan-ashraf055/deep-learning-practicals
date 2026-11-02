
"""
Forward Propagation in a Multilayer Perceptron (MLP)
Part A: the 2-2-1 worked example from class, neuron by neuron and as matrices.
Part B: a general forward pass for ANY layer sizes, with shapes and parameter count.

"""

import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
relu = lambda z: np.maximum(0, z)

# ---------------- Part A: the 2-2-1 example ----------------
x  = np.array([1.0, 0.5])
W1 = np.array([[0.5, -0.4],
               [0.3,  0.8]]);  b1 = np.array([0.1, -0.2])
W2 = np.array([[1.2, -0.7]]);   b2 = np.array([0.3])

print("Part A: neuron by neuron")
z_h1 = 0.5*1.0 + (-0.4)*0.5 + 0.1
z_h2 = 0.3*1.0 + 0.8*0.5 - 0.2
a_h1, a_h2 = sigmoid(z_h1), sigmoid(z_h2)
z_out = 1.2*a_h1 - 0.7*a_h2 + 0.3
print(f"  h1: z = {z_h1:.4f}, a = {a_h1:.4f}")
print(f"  h2: z = {z_h2:.4f}, a = {a_h2:.4f}")
print(f"  output: z = {z_out:.4f}, y_hat = {sigmoid(z_out):.4f}")

print("\nPart A: same thing with matrices")
z1 = W1 @ x + b1;  a1 = sigmoid(z1)
z2 = W2 @ a1 + b2; a2 = sigmoid(z2)
print("  z1 =", z1.round(4), " a1 =", a1.round(4))
print("  z2 =", z2.round(4), " y_hat =", a2.round(4))
print("  Prediction: class", int(a2[0] > 0.5), f"({a2[0]:.0%} confident)")

# ---------------- Part B: any network ----------------
def init_network(layer_sizes, seed=0):
    """Random weights W (n_out x n_in) and zero biases for each layer."""
    rng = np.random.default_rng(seed)
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.normal(0, np.sqrt(2 / n_in), (n_out, n_in))
        b = np.zeros(n_out)
        params.append((W, b))
    return params

def forward(X, params):
    """X has one example per ROW. ReLU in hidden layers, softmax at the end."""
    A = X
    for i, (W, b) in enumerate(params):
        Z = A @ W.T + b
        if i < len(params) - 1:
            A = relu(Z)
        else:
            E = np.exp(Z - Z.max(axis=1, keepdims=True))
            A = E / E.sum(axis=1, keepdims=True)
        print(f"  layer {i+1}: W {W.shape}, output {A.shape}")
    return A

sizes = [784, 128, 64, 10]                # the MNIST network from class
params = init_network(sizes)
n_params = sum(W.size + b.size for W, b in params)
print(f"\nPart B: network {sizes}")
for i, (W, b) in enumerate(params):
    print(f"  layer {i+1}: {W.size:,} weights + {b.size} biases = {W.size + b.size:,}")
print(f"  total parameters = {n_params:,}")

print("\nForward pass on a batch of 32 random 'images':")
X_batch = np.random.default_rng(1).random((32, 784))
probs = forward(X_batch, params)
print("  each row sums to 1:", np.allclose(probs.sum(axis=1), 1))
print("  predicted digits (untrained, so just guesses):", probs.argmax(axis=1)[:10])