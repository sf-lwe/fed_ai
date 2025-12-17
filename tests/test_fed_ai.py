"""Unit tests for fed_ai components."""

import numpy as np
import pytest
from fed_ai.task import TrainingProgressCallback, training_callbacks
from fed_ai.client_app import FlowerClient


def test_preprocessing_returns_arrays():
    """Test that FlowerClient._preprocess_images returns numpy arrays."""
    # Mock data
    data = (
        np.random.rand(10, 120, 120, 3).astype('float32'),
        np.random.randint(0, 10, 10),
        np.random.rand(5, 120, 120, 3).astype('float32'),
        np.random.randint(0, 10, 5)
    )
    client = FlowerClient(0.001, data, 1, 8, None)
    assert isinstance(client.x_train, np.ndarray)
    assert isinstance(client.y_train, np.ndarray)
    assert isinstance(client.x_test, np.ndarray)
    assert isinstance(client.y_test, np.ndarray)
    assert client.x_train.shape == (10, 120, 120, 3)
    assert client.y_train.shape == (10,)


def test_training_callback_instantiation():
    """Test that TrainingProgressCallback can be instantiated."""
    callback = TrainingProgressCallback()
    assert callback is not None
    assert hasattr(callback, 'on_train_begin')


def test_training_callbacks_function():
    """Test that training_callbacks returns a list with the callback."""
    callbacks = training_callbacks()
    assert isinstance(callbacks, list)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], TrainingProgressCallback)