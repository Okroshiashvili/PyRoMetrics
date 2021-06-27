from typing import Union

import pandas as pd
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


def _geometric_mean(a: np.ndarray, axis: int = 0, dtype=None):
    """
    Calculates geometric average of a series

    Returns:
        Geometric average
    """

    if not isinstance(a, np.ndarray):
        log_a = np.log(np.array(a, dtype=dtype))
    else:
        log_a = np.log(a)

    return np.exp(log_a.mean(axis=axis))


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """
    Naive forecasting method that repeats previous value

    Args:
        actual (np.ndarray): array of actual values
        seasonality (int, optional): Defaults to 1.

    Returns:
        Naive forecast
    """

    return actual[:-seasonality]


def _relative_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: Union[np.ndarray, int] = None,
):
    """
    Calculates error relative to provided benchmark

    Args:
        actual (np.ndarray): Actual value
        predicted (np.ndarray): Predicted value
        benchmark (Union[np.ndarray, int], optional): User provided benchmark

    Raises:
        ValueError if benchmark is not instance of either integer, Numpy array or Pandas series

    Returns:
        Relative error
    """

    error = 0

    if isinstance(benchmark, (np.ndarray, pd.Series)):
        error = _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)

    elif isinstance(benchmark, int):
        error = _error(actual[benchmark:], predicted[benchmark:]) / (
            _error(actual[benchmark:], _naive_forecasting(actual, benchmark)) + EPSILON
        )

    else:
        raise ValueError(
            "Benchmark should be either integer, Numpy Ndarray or Pandas Series"
        )

    return error


def _bounded_relative_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: Union[np.ndarray, int] = None,
):
    """
    Calculates bounded error relative to provided benchmark

    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        benchmark (Union[np.ndarray, int], optional): User provided benchmark.

    Raises:
        ValueError if benchmark is not instance of either integer, Numpy array or Pandas series

    Returns:
        Bounded Relative Error
    """

    abs_error = 0
    abs_error_benchmark = 0

    if isinstance(benchmark, (np.ndarray, pd.Series)):
        abs_error = np.abs(_error(actual, predicted))
        abs_error_benchmark = np.abs(_error(actual, benchmark))

    elif isinstance(benchmark, int):
        abs_error = np.abs(_error(actual[benchmark:], predicted[benchmark:]))
        abs_error_benchmark = np.abs(
            _error(actual[benchmark:], _naive_forecasting(actual, benchmark))
        )

    else:
        raise ValueError(
            "Benchmark should be either integer, Numpy Ndarray or Pandas Series"
        )

    return abs_error / (abs_error + abs_error_benchmark + EPSILON)
