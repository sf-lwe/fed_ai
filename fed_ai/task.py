import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
import logging
import time

from keras.callbacks import Callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(120, 120, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="marmal88/skin_cancer",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["image"] / 255.0, partition["train"]["dx"]
    x_test, y_test = partition["test"]["image"] / 255.0, partition["test"]["dx"]
    logger.info("Currently loading data")
    return x_train, y_train, x_test, y_test


class TrainingProgressCallback(Callback):
    """Keras Callback that logs training progress (epoch, batch, loss, metrics).

    This prints concise messages so you can see whether the model is training
    and what the current step/epoch is.
    """

    def __init__(self, logger=None, log_every_n_batches: int = 10):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.log_every_n_batches = max(1, int(log_every_n_batches))
        self.current_epoch = None

    def on_train_begin(self, logs=None):
        # params should include 'epochs' and 'steps'
        epochs = self.params.get("epochs")
        steps = self.params.get("steps")
        self.logger.info(
            f"Training starting: epochs={epochs}, steps_per_epoch={steps}"
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch+1}/{self.params.get('epochs')} started")

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        # logs may contain loss and other metrics
        dur = time.time() - getattr(self, "batch_start_time", time.time())
        loss = logs.get("loss")
        # Show a short summary every N batches to avoid noisy output
        if (batch + 1) % self.log_every_n_batches == 0:
            metrics = {k: v for k, v in logs.items() if k != "loss"}
            self.logger.info(
                (
                    f"Epoch {self.current_epoch+1}/{self.params.get('epochs')}: "
                    f"batch {batch+1}/{self.params.get('steps')}, "
                    f"loss={loss:.4f}, metrics={metrics}, batch_time={dur:.3f}s"
                )
            )

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        metrics = {k: v for k, v in logs.items() if k != "loss"}
        self.logger.info(
            f"Epoch {epoch+1}/{self.params.get('epochs')} finished. loss={loss}, metrics={metrics}"
        )


def training_callbacks(log_every_n_batches: int = 10):
    """Return a list of callbacks to pass into model.fit for progress logging."""

    return [TrainingProgressCallback(logger=logger, log_every_n_batches=log_every_n_batches)]
