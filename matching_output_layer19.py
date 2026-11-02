
"""
Matching the Output Layer and Loss to the Task (Keras)
Three tiny models on synthetic data:
  regression  -> 1 linear neuron  + MSE
  binary      -> 1 sigmoid neuron + binary cross-entropy
  multi-class -> K softmax neurons + (sparse) categorical cross-entropy
Run in Google Colab.

"""

import numpy as np
from tensorflow import keras

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (2000, 4)).astype("float32")

def mlp(output_units, output_activation):
    return keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(output_units, activation=output_activation),
    ])

# ---- 1. Regression: predict a number ----
y_reg = (3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] * X[:, 3]).astype("float32")
reg = mlp(1, None)                                   # None = linear
reg.compile(optimizer="adam", loss="mse", metrics=["mae"])
reg.fit(X, y_reg, epochs=30, batch_size=32, verbose=0)
print("Regression   MSE, MAE:", np.round(reg.evaluate(X, y_reg, verbose=0), 3))

# ---- 2. Binary classification: yes / no ----
y_bin = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype("float32")   # inside a circle?
binary = mlp(1, "sigmoid")
binary.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
binary.fit(X, y_bin, epochs=30, batch_size=32, verbose=0)
print("Binary       loss, accuracy:", np.round(binary.evaluate(X, y_bin, verbose=0), 3))

# ---- 3. Multi-class: one of 3 classes ----
y_cls = np.digitize(X[:, 0] + X[:, 1], [-1, 1])               # labels 0, 1, 2
multi = mlp(3, "softmax")
multi.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
multi.fit(X, y_cls, epochs=30, batch_size=32, verbose=0)
print("Multi-class  loss, accuracy:", np.round(multi.evaluate(X, y_cls, verbose=0), 3))

# The same labels one-hot encoded need categorical_crossentropy instead:
Y_onehot = keras.utils.to_categorical(y_cls, 3)
print("one-hot label example:", y_cls[0], "->", Y_onehot[0])
print("probabilities for 2 examples:\n", multi.predict(X[:2], verbose=0).round(3))