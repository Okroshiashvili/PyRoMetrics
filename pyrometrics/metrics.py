import numpy as np

from pyrometrics.helpers import _error, _percentage_error, _geometric_mean, EPSILON


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


def mdae(actual: np.ndarray, predicted: np.ndarray):
    """
    MDAE - Median Absolute Error

    This is median of absolute value of error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Median Absolute Error
    """

    return np.median(np.abs(_error(actual, predicted)))


def gmae(actual: np.ndarray, predicted: np.ndarray):
    """
    GMAE - Geometric Mean Absolute Error

    This is a geometric mean of absolute value of error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Geometric Mean Absolute Error
    """

    return _geometric_mean(np.abs(_error(actual, predicted)))


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


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """
    NRMSE - Normalized Root Mean Square Error

    This is normalized version of root mean square error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Normalized Root Mean Square Error
    """
    return rmse(actual, predicted) / (actual.max() - actual.min())


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


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    SMAPE - Symmetric Mean Absolute Percentage Error

    This is symmetric version of mean of absolute value of percentage error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Symmetric Mean Absolute Percentage Error
    """

    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )
