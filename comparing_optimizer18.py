
"""
Comparing Optimizers in Keras: SGD vs Momentum vs RMSProp vs Adam
Trains the same MNIST MLP four times and plots the loss curves.
Run in Google Colab. Takes a few minutes on CPU, faster with a GPU runtime.

"""

import matplotlib.pyplot as plt
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

def build_model():
    return keras.Sequential([
        keras.Input(shape=(784,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

optimizers = {
    "SGD (0.01)":             lambda: keras.optimizers.SGD(learning_rate=0.01),
    "SGD + momentum (0.9)":   lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSProp (0.001)":        lambda: keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam (0.001)":           lambda: keras.optimizers.Adam(learning_rate=0.001),
}

keras.utils.set_random_seed(42)          # same starting weights each time
results = {}
for name, make_opt in optimizers.items():
    keras.utils.set_random_seed(42)
    model = build_model()
    model.compile(optimizer=make_opt(),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    results[name] = hist.history["loss"]
    print(f"{name:22s} final train loss {hist.history['loss'][-1]:.4f}   test accuracy {acc:.4f}")

for name, losses in results.items():
    plt.plot(range(1, 6), losses, marker="o", label=name)
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Same network, different optimizers")
plt.show()