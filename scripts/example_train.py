"""Quick example to exercise the training progress callback.

This script creates a model from `fed_ai.task.load_model`, generates
random data, and runs `model.fit(...)` with the `training_callbacks` so you
can see epoch and batch-level logging locally.
"""
import numpy as np
from fed_ai.task import load_model, training_callbacks


def main():
    model = load_model(learning_rate=0.001)

    # Create small random dataset: 32 samples, images 120x120x3, 10 classes
    x = np.random.rand(32, 120, 120, 3).astype("float32")
    y = np.random.randint(0, 10, size=(32,)).astype("int32")

    callbacks = training_callbacks(log_every_n_batches=4)

    print("Starting example training (2 epochs) — you should see callback logs below")
    model.fit(x, y, epochs=2, batch_size=8, verbose=0, callbacks=callbacks)
    print("Example training finished")


if __name__ == "__main__":
    main()
