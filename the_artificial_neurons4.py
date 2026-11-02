
"""
The Artificial Neuron (McCulloch-Pitts style)
A neuron = weighted sum + bias + activation. Here the weights are SET BY HAND
to make logic gates. (Program 5 makes the neuron LEARN them.)
"""


import numpy as np

def step(z):
    return 1 if z > 0 else 0

def neuron(x, w, b):
    z = np.dot(w, x) + b          # weighted sum
    return step(z)                # activation

gates = {
    "AND":  (np.array([1, 1]),   -1.5),
    "OR":   (np.array([1, 1]),   -0.5),
    "NAND": (np.array([-1, -1]),  1.5),
    "NOR":  (np.array([-1, -1]),  0.5),
}
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print(" x1 x2 | " + "  ".join(f"{g:>4s}" for g in gates))
for x in inputs:
    outs = [neuron(np.array(x), w, b) for w, b in gates.values()]
    print(f"  {x[0]}  {x[1]} | " + "  ".join(f"{o:>4d}" for o in outs))

# ---- Intuition example from class: go to the cricket match? ----
w = np.array([3, 2, -4])   # good weather, friend coming, exam tomorrow
b = -2                     # "a bit lazy by default"
for x in [(1, 1, 0), (1, 1, 1)]:
    z = np.dot(w, x) + b
    print(f"\nInputs {x}: z = {z} -> {'GO!' if step(z) else 'Stay home and study'}")