"""tfexample: A Flower / TensorFlow app."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed_ai.task import load_data, load_model, training_callbacks
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        verbose,
    ):
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_test, self.y_test = self._preprocess_images(
            data
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        logger.info(self.x_train.shape)
        logger.info(self.x_test.shape)
        self.model.set_weights(parameters)
        # Use a training callback to log progress (epoch and batch information)
        # If verbose is provided in the run config, allow Keras to use it; otherwise
        # set verbose=0 so logs come only from the callback.
        verbose_arg = self.verbose if self.verbose is not None else 0
        callbacks = training_callbacks(log_every_n_batches=10)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=verbose_arg,
            callbacks=callbacks,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

    def _preprocess_images(self, data):
        # `data` is expected to be a partition dict returned by `load_data`.
        # The previous implementation returned a generator which caused Keras
        # to receive a non-array object. Convert images to a single NumPy
        # array for train and test and return the four arrays.
        #
        # `data` from load_data() is (x_train, y_train, x_test, y_test)
        try:
            x_train, y_train, x_test, y_test = data
        except Exception:
            # If data is already a dataset-like object, let the error surface
            raise ValueError("`data` must be a 4-tuple: (x_train, y_train, x_test, y_test)")

        def _resize(images):
            # Accept numpy arrays or lists; convert to tensor then resize
            t = tf.convert_to_tensor(images)
            resized = tf.image.resize(t, (120, 120)).numpy()
            return resized

        x_train = _resize(x_train)
        x_test = _resize(x_test)

        return x_train, y_train, x_test, y_test


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    print(data)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
