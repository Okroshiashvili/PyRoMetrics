import numpy as np

from pyrometrics.helpers import _error, _percentage_error


def mse(actual: np.ndarray, predicted: np.ndarray):
    """
    MSE - Mean Square Error

    This is an arithmetic average of squared error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Mean Square Error
    """

    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """
    RMSE - Root Mean Square Error

    This is a square root from mean of squared error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Root Mean Square Error
    """

    return np.sqrt(mse(actual, predicted))


def me(actual: np.ndarray, predicted: np.ndarray):
    """
    ME - Mean Error

    This is an arithmetic average of error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Mean Error
    """

    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """
    MAE - Mean Absolute Error

    This is an arithmetic average of absolute value of error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Mean Absolute Error
    """

    return np.mean(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """
    MPE - Mean Percentage Error

    This is an arithmetic average of percentage error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Mean Percentage Error
    """

    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    MAPE - Mean Absolute Percentage Error

    This is an arithmetic average of absolute value of percentage error

    Note: The result is not multiplied by 100

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Mean Absolute Percentage Error
    """

    return np.mean(np.abs(_percentage_error(actual, predicted)))
