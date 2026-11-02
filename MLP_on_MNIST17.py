
"""
MLP on MNIST with Keras (TensorFlow)
The same 784-128-64-10 network from class, trained the practical way.
Run in Google Colab (TensorFlow is pre-installed).

"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# ---- 1. Load and prepare the data ----
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("train:", x_train.shape, "test:", x_test.shape)        # (60000, 28, 28)

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0  # flatten + scale
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

# ---- 2. Build the model (Session 5) ----
model = keras.Sequential([
    keras.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.summary()                  # should show 109,386 parameters

# ---- 3. Loss + optimizer (Sessions 6-7) ----
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",   # labels are 0-9
              metrics=["accuracy"])

# ---- 4. Train with mini-batches (Session 6) ----
history = model.fit(x_train, y_train,
                    batch_size=32, epochs=10,
                    validation_split=0.1)

# ---- 5. Evaluate ----
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# ---- 6. Learning curves ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.show()

# ---- 7. Predictions = forward propagation (Session 5) ----
probs = model.predict(x_test[:10], verbose=0)
preds = probs.argmax(axis=1)
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, a in enumerate(axes):
    a.imshow(x_test[i].reshape(28, 28), cmap="gray_r"); a.axis("off")
    a.set_title(f"{preds[i]} ({probs[i].max():.0%})",
                color="green" if preds[i] == y_test[i] else "red")
plt.show()