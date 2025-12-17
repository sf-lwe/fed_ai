"""Streamlit app for visualizing federated AI training progress.

Run with: streamlit run streamlit_app.py

This app simulates a federated learning run with multiple clients,
showing client selection, training progress, and sample data from a specific client.
"""

import streamlit as st
import numpy as np
import pandas as pd
from fed_ai.task import load_model
import time
from keras.callbacks import Callback
import random
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import tensorflow as tf


# Generate client names and colors
def generate_client_info(num_clients):
    names = [
        "The Spot Stop",
        "Derm & Giggles",
        "Pimple Palace",
        "Glow & Tell Clinic",
        "Doc in a Box",
        "The Rash Decision",
        "Skincredible Care",
        "The Healing Giggles",
        "Oopsie Daisy Dermatology",
        "Freckle Fixers",
        "Smooth Operators Clinic",
        "Paging Dr. Cute",
        "The Boo-Boo Bureau",
        "Acne Ain't Clinic",
        "Snip Snip Hooray",
        "The Happy Derm",
        "Glow Patrol",
        "The Itch Relief Society",
        "Dermy & Friends",
        "All Better Now Clinic",
    ]

    colors = [
        "#F63366",  # Streamlit Accent Pink
        "#4C72B0",  # Soft Blue
        "#55A868",  # Soft Green
        "#C44E52",  # Muted Red
        "#8172B2",  # Indigo-Purple
        "#CCB974",  # Warm Khaki
        "#64B5F6",  # Light Blue
        "#90A4AE",  # Blue-Gray
        "#7CB342",  # Fresh Green
        "#E57373",  # Muted Coral
        "#BA68C8",  # Soft Purple
        "#AED581",  # Sage Green
        "#FFD54F",  # Soft Yellow
        "#FFB74D",  # Apricot
        "#4DB6AC",  # Aqua
        "#7986CB",  # Indigo
        "#81C784",  # Mint Green
        "#A1887F",  # Warm Taupe
        "#9575CD",  # Lavender
        "#8D6E63",  # Earth Brown
    ]

    client_names = random.sample(names, num_clients)
    client_colors = random.sample(colors, num_clients)
    return client_names, client_colors


class StreamlitTrainingCallback(Callback):
    """Modified callback that updates Streamlit for visualization."""

    def __init__(
        self, client_id, round_num, logger=None, log_every_n_batches: int = 10
    ):
        self.client_id = client_id
        self.round_num = round_num
        self.logger = logger
        self.log_every_n_batches = max(1, int(log_every_n_batches))
        self.current_epoch = None
        self.metrics_history = []

    def on_train_begin(self, logs=None):
        epochs = self.params.get("epochs")
        steps = self.params.get("steps")
        st.write(
            f"Client {self.client_id}, Round {self.round_num}: Training starting: epochs={epochs}, steps_per_epoch={steps}"
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        st.write(
            f"Client {self.client_id}, Round {self.round_num}: Epoch {epoch + 1}/{self.params.get('epochs')} started"
        )

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        dur = time.time() - getattr(self, "batch_start_time", time.time())
        loss = logs.get("loss")
        accuracy = logs.get("accuracy", 0.0)
        self.metrics_history.append(
            {
                "client": self.client_id,
                "round": self.round_num,
                "epoch": self.current_epoch + 1,
                "batch": batch + 1,
                "loss": loss,
                "accuracy": accuracy,
                "batch_time": dur,
            }
        )
        if (batch + 1) % self.log_every_n_batches == 0:
            st.write(
                f"Client {self.client_id}, Round {self.round_num}: Epoch {self.current_epoch + 1}/{self.params.get('epochs')}: batch {batch + 1}/{self.params.get('steps')}, loss={loss:.4f}, accuracy={accuracy:.4f}, batch_time={dur:.3f}s"
            )

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        accuracy = logs.get("accuracy", 0.0)
        st.write(
            f"Client {self.client_id}, Round {self.round_num}: Epoch {epoch + 1}/{self.params.get('epochs')} finished. loss={loss:.4f}, accuracy={accuracy:.4f}"
        )


def load_real_data_sample(
    client_id=0, num_partitions=10, num_samples=20
) -> tuple[np.ndarray, np.ndarray]:
    """Load a sample of real data from the dataset for a specific client."""
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="marmal88/skin_cancer",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(client_id, "train")
    partition.set_format("numpy")

    # Take first num_samples (or all if less)
    total_samples = len(partition)
    num_to_load = min(num_samples, total_samples)
    images = partition["image"][:num_to_load] / 255.0  # Normalize
    labels = partition["dx"][:num_to_load]

    # Resize to 120x120 as in the model
    resized_images = []
    for img in images:
        resized = tf.image.resize(tf.convert_to_tensor(img), (120, 120)).numpy()
        resized_images.append(resized)

    return np.array(resized_images), labels


def simulate_federated_learning(
    num_clients=10,
    num_rounds=3,
    fraction_fit=0.5,
    colors="#000000",
    local_epochs=1,
    batch_size=8,
):
    """Simulate FL with multiple clients."""
    client_names, _ = generate_client_info(num_clients)

    # Initialize global model
    global_model = load_model(learning_rate=0.001)
    global_weights = global_model.get_weights()

    all_metrics = []

    for round_num in range(1, num_rounds + 1):
        st.subheader(f"Round {round_num}")

        # Select clients
        num_selected = int(num_clients * fraction_fit)
        selected_clients = random.sample(range(num_clients), num_selected)
        selected_names = [client_names[i] for i in selected_clients]
        st.write(f"Selected clients: {selected_names}")

        client_weights = []

        for client_id in selected_clients:
            st.write(f"Training {client_names[client_id]} (ID: {client_id})")

            # Generate client data (random for simulation)
            x = np.random.rand(32, 120, 120, 3).astype("float32")
            y = np.random.randint(0, 10, size=(32,)).astype("int32")

            # Load client model with global weights
            client_model = load_model(learning_rate=0.001)
            client_model.set_weights(global_weights)

            callback = StreamlitTrainingCallback(
                client_id, round_num, log_every_n_batches=4
            )
            client_model.fit(
                x,
                y,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[callback],
            )

            client_weights.append(client_model.get_weights())
            all_metrics.extend(callback.metrics_history)

        # Simple averaging for aggregation
        new_weights = []
        for layer in range(len(global_weights)):
            layer_weights = [w[layer] for w in client_weights]
            new_weights.append(np.mean(layer_weights, axis=0))
        global_weights = new_weights

        st.write(f"Round {round_num} aggregation completed.")

    return all_metrics


def main():
    st.set_page_config(layout="wide")

    st.title(
        "Federated AI Training Progress Visualization",
        width="stretch",
        text_alignment="center",
    )

    # Create two columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col3:
        st.subheader("Simulation Settings")
        num_clients = st.slider("Number of Clients", 5, 20, 10)
        num_rounds = st.slider("Number of Rounds", 1, 20, 3)
        fraction_fit = st.slider("Fraction of Clients per Round", 0.1, 1.0, 0.5)

        # Cache client names and colors in session state
        if (
            "num_clients_cached" not in st.session_state
            or st.session_state.num_clients_cached != num_clients
        ):
            st.session_state.num_clients_cached = num_clients
            st.session_state.client_names, st.session_state.client_colors = (
                generate_client_info(num_clients)
            )

        client_names = st.session_state.client_names
        client_colors = st.session_state.client_colors

    with col1:

        st.subheader("Client Data Visualization", text_alignment="center")
        for i_client in range(num_clients):
            st.button(client_names[i_client], width="stretch")

    with col2:
        st.subheader("Federated Learning Simulation", text_alignment="center")
        if st.button("Start Federated Simulation", width="stretch"):
            st.write("Starting federated learning simulation...")
            with st.container(height=500):
                metrics = simulate_federated_learning(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    fraction_fit=fraction_fit,
                    colors=client_colors,
                    local_epochs=1,
                    batch_size=8,
                )
            with st.container():
                if metrics:
                    df = pd.DataFrame(metrics)
                    st.subheader("Aggregated Loss Over Rounds")
                    round_loss = df.groupby("round")["loss"].mean()
                    st.line_chart(round_loss)

                    st.subheader("Aggregated Accuracy Over Rounds")
                    round_acc = df.groupby("round")["accuracy"].mean()
                    st.line_chart(round_acc)


if __name__ == "__main__":
    main()
