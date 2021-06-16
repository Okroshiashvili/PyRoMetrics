import numpy as np


EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculates differene between actual and predicted values.
    This is the model error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Array of difference between actual and predicted
    """

    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculates percentage error. Note that, in denominator there is EPSILON,
    very low number to avoid zero devision.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Percentage error NOT multiplied by 100
    """

    return _error(actual, predicted) / (actual + EPSILON)


def _geometric_mean(a, axis=0, dtype=None):

    if not isinstance(a, np.ndarray):
        log_a = np.log(np.array(a, dtype=dtype))
    else:
        log_a = np.log(a)

    return np.exp(log_a.mean(axis=axis))
