from typing import Union

import numpy as np

from pyrometrics._helpers import (
    _error,
    _percentage_error,
    _geometric_mean,
    _naive_forecasting,
    _relative_error,
    _bounded_relative_error,
    _EPSILON,
)


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


def nae(actual: np.ndarray, predicted: np.ndarray):
    """
    NAE - Normalized Absolute Error

    This is normalized version of mean of absolute error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Normalized Absolute Error
    """

    __mae = mae(actual, predicted)

    return np.sqrt(
        np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1)
    )


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonality: int = 1,
):
    """
    MASE - Mean Absolute Scaled Error

    This is a scaled version of mean absolute error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        seasonality (int, optional): User provided seasonality for calculating benchmark forecast

    Returns:
        Mean Absolute Scaled Error
    """

    return mae(actual, predicted) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )


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


def rmsse(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonality: int = 1,
):
    """
    RMSSE - Root Mean Square Scaled Error

    This is scaled version of root from the mean squared error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        seasonality (int, optional): User provided seasonality

    Returns:
        Root Mean Square Scaled Error
    """

    q = np.abs(_error(actual, predicted)) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )

    return np.sqrt(np.mean(np.square(q)))


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


def inrse(actual: np.ndarray, predicted: np.ndarray):
    """
    INRSE - Integral Normalized Root Squared Error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Integral Normalized Root Squared Error
    """

    return np.sqrt(
        np.sum(np.square(_error(actual, predicted)))
        / np.sum(np.square(actual - np.mean(actual)))
    )


# Percentage Errors


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """
    MPE - Mean Percentage Error

    This is an arithmetic average of percentage error

    Note: The result is not multiplied by 100

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

    This is symmetric version of mean absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Symmetric Mean Absolute Percentage Error
    """

    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + _EPSILON)
    )


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """
    MDAPE - Median Absolute Percentage Error

    Median value of absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Median Absolute Percentage Error
    """

    return np.median(np.abs(_percentage_error(actual, predicted)))


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """
    SMDAPE - Symmetric Median Absolute Percentage Error

    This is a symmetric version of median absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Symmetric Median Absolute Percentage Error
    """

    return np.median(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + _EPSILON)
    )


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    MAAPE - Mean Arctangent Absolute Percentage Error

    This is mean of arctangent of absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Mean Arctangent Absolute Percentage Error
    """

    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + _EPSILON))))


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    RMSPE - Root Mean Squared Percentage Error

    This is a square root from the mean of the percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Root Mean Squared Percentage Error
    """

    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    RMDSPE - Root Median Squared Percentage Error

    This is square root from the meadian value of squared percentage error

    Note: The result is not multiplied by 100

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Root Median Squared Percentage Error
    """

    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def nape(actual: np.ndarray, predicted: np.ndarray):
    """
    NAPE - Normalized Absolute Percentage Error

    This is normalized version of absolute percentage error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Normaized Absolute Percentage Error
    """

    __mape = mape(actual, predicted)

    return np.sqrt(
        np.sum(np.square(_percentage_error(actual, predicted) - __mape))
        / (len(actual) - 1)
    )


# Relative Errors


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: Union[np.ndarray, int]):
    """
    MRE - Mean Relative Error

    This is mean of relative error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        benchmark (Union[np.ndarray, int]): array of benchmark values

    Returns:
        Mean Relative Error
    """

    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """
    RAE - Relative Absolute Error

    A.K.A - Approximation Error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Relative Absolute Error
    """

    return np.sum(np.abs(actual - predicted)) / (
        np.sum(np.abs(actual - np.mean(actual))) + _EPSILON
    )


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: Union[np.ndarray, int]):
    """
    MRAE - Mean Relative Absolute Error

    This is the mean of relative absolute error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        benchmark (Union[np.ndarray, int]): array of benchmark values

    Returns:
        Mean Relative Absolute Error
    """

    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: Union[np.ndarray, int],
):
    """
    MDRAE - Median Relative Absolute Error

    This is median of relative absolute error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        benchmark (Union[np.ndarray, int]): array of benchmark values

    Returns:
        Median Relative Absolute Error
    """

    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """
    RRSE - Root Relative squared Error

    This is square root from relative squared error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Root Relative Squared Error
    """

    return np.sqrt(
        np.sum(np.square(actual - predicted))
        / np.sum(np.square(actual - np.mean(actual)))
    )


def gmrae(
    actual: np.ndarray, predicted: np.ndarray, benchmark: Union[np.ndarray, int] = None
):
    """
    GMRAE - Geometric Mean Relative Absolute Error

    This is the geometric mean of relative absolute error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        benchmark (Union[np.ndarray, int]): array of benchmark values

    Returns:
        Geometric Mean Relative Absolute Error
    """

    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: Union[np.ndarray, int] = None,
):
    """
    MBRAE - Mean Bounded Relative Absolute Error

    This is mean of bounded relative absolute error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        benchmark (Union[np.ndarray, int]): array of benchmark values

    Returns:
        Mean Bounded Relative Absolute Error
    """

    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: Union[np.ndarray, int] = None,
):
    """
    UMBRAE - Unscaled Mean Bounded Relative Absolute Error

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        benchmark (Union[np.ndarray, int]): array of benchmark values

    Returns:
        Unscaled Mean Bounded Relative Absolute Error
    """

    __mbrae = mbrae(actual, predicted, benchmark)

    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """
    MDA - Mean Directional Accuracy

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values

    Returns:
        Mean Directional Accuracy
    """

    return np.mean(
        (
            np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])
        ).astype(int)
    )
