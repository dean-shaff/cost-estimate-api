# cost_estimate_linear_regression.py
import numpy as np

ArrayType = np.ndarray


__all__ = [
    "cost_estimate_linear_regression"
]


def lstsq(x: ArrayType, y: ArrayType):
    """
    Compute least square solution, using np.linalg.lstsq
    """
    # first we have to append offset column to x.
    A = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    weights = np.linalg.lstsq(A, y)[0]
    return weights


def cost_estimate_linear_regression(x: ArrayType, y: ArrayType) -> ArrayType:
    """
    Given some training data, train a neural network for linear regression, returning the corresponding weights.

    Args:
        train_data: The first column of the array should be the costs,
            while the remaining columns correspond to test samples.

    Returns:
        ArrayType: weights, as computed by neural network.
    """
    weights = lstsq(x, y)
    return weights
