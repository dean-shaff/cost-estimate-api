# cost_estimate_linear_regression.py
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from .types import ArrayType

module_logger = logging.getLogger(__name__)


__all__ = [
    "lstsq",
    "create_tf_model",
    "compute_weights_lstsq",
    "compute_weights_tf"
]


def lstsq(x: ArrayType, y: ArrayType):
    """
    Compute least square solution, using np.linalg.lstsq

    Args:
        x: input data
        y: target data
    Returns:
        ArrayType: array containing weights
    """
    # first we have to append offset column to x.
    A = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    weights = np.linalg.lstsq(A, y, rcond=None)[0]
    return weights


def create_tf_model(n_features: int, **kwargs) -> tf.keras.Model:
    """
    Create a simple Tensorflow model for doing linear regression.

    Args:
        n_features: The number of variables in the regression, not including bias;
            Tensorflow automatically adds bias.
        kwargs: keyword arguments passed to tf.Model.compile method.
    Returns:
        tf.Model: Tensorflow model corresponding to a single layer neural network without normalization.
    """
    model = tf.keras.Sequential([
        layers.Dense(units=1, input_shape=[n_features, ])
    ])
    compile_kwargs = dict(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error'
    )

    compile_kwargs.update(kwargs)

    model.compile(**compile_kwargs)

    return model


def compute_weights_lstsq(x: ArrayType, y: ArrayType, **kwargs) -> ArrayType:
    """
    Given some training data, get least squares solution to linear regression problem.

    Example:

        >>> x.shape
        (10, 4)
        >>> y.shape
        (10, )
        >>> compute_weights_lstsq(x, y)
        array([0, 0.56, 5.6, 10.0, -0.1])

    Args:
        x: input data
        y: target data

    Returns:
        ArrayType: weights, as computed by numpy.linalg.lstsq

    """
    return lstsq(x, y).reshape((-1, ))


def compute_weights_tf(x: ArrayType, y: ArrayType, learning_rate=0.001, epochs=400) -> ArrayType:
    """
    Given some training data, train a neural network for linear regression, returning the corresponding weights.

    Example:

        >>> x.shape
        (10, 4)
        >>> y.shape
        (10, )
        >>> compute_weights_tf(x, y)
        array([0, 0.56, 5.6, 10.0, -0.1])

    Args:
        x: input data
        y: target data

    Returns:
        ArrayType: weights, as computed by neural network.
    """
    module_logger.debug(f"compute_weights_tf: learning_rate={learning_rate}")
    module_logger.debug(f"compute_weights_tf: epochs={epochs}")
    # weights = lstsq(x, y)
    n_features = x.shape[1]
    model = create_tf_model(n_features, optimizer=tf.optimizers.Adam(learning_rate=learning_rate))
    model.fit(x, y, epochs=epochs, verbose=0)

    # we have to extract weights and bias seperately from model
    weights = np.zeros(n_features + 1)
    weights[:x.shape[1]] = (model.layers[0].weights[0].numpy())[:, 0]
    weights[-1] = model.layers[0].weights[1].numpy()

    return weights
