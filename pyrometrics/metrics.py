from typing import Union

import pandas as pd
import numpy as np

from pyrometrics.helpers import (
    _error,
    _percentage_error,
    _geometric_mean,
    _naive_forecasting,
    _relative_error,
    _bounded_relative_error,
    EPSILON,
)


def me(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    ME - Mean Error

    This is an arithmetic average of error

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Mean Error
    """

    return np.mean(_error(actual, predicted))


def mae(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    MAE - Mean Absolute Error

    This is an arithmetic average of absolute value of error

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Mean Absolute Error
    """

    return np.mean(np.abs(_error(actual, predicted)))


def nae(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    NAE - Normalized Absolute Error

    This is normalized version of mean of absolute error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Normalized Absolute Error
    """

    __mae = mae(actual, predicted)

    return np.sqrt(
        np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1)
    )


def mase(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    seasonality: int = 1,
):
    """
    MASE - Mean Absolute Scaled Error

    This is a scaled version of mean absolute error

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values
        seasonality (int, optional): User provided seasonality for calculating benchmark forecast

    Returns:
        Mean Absolute Scaled Error
    """

    return mae(actual, predicted) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )


def mdae(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    MDAE - Median Absolute Error

    This is median of absolute value of error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Median Absolute Error
    """

    return np.median(np.abs(_error(actual, predicted)))


def gmae(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    GMAE - Geometric Mean Absolute Error

    This is a geometric mean of absolute value of error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Geometric Mean Absolute Error
    """

    return _geometric_mean(np.abs(_error(actual, predicted)))


def mse(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    MSE - Mean Square Error

    This is an arithmetic average of squared error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Mean Square Error
    """

    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    RMSE - Root Mean Square Error

    This is a square root from mean of squared error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Root Mean Square Error
    """

    return np.sqrt(mse(actual, predicted))


def rmsse(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    seasonality: int = 1,
):
    """
    RMSSE - Root Mean Square Scaled Error

    This is scaled version of root from the mean squared error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values
        seasonality (int, optional): User provided seasonality

    Returns:
        Root Mean Square Scaled Error
    """

    q = np.abs(_error(actual, predicted)) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )

    return np.sqrt(np.mean(np.square(q)))


def nrmse(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    NRMSE - Normalized Root Mean Square Error

    This is normalized version of root mean square error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Normalized Root Mean Square Error
    """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def inrse(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    INRSE - Integral Normalized Root Squared Error

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Integral Normalized Root Squared Error
    """

    return np.sqrt(
        np.sum(np.square(_error(actual, predicted)))
        / np.sum(np.square(actual - np.mean(actual)))
    )


# Percentage Errors


def mpe(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    MPE - Mean Percentage Error

    This is an arithmetic average of percentage error

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Mean Percentage Error
    """

    return np.mean(_percentage_error(actual, predicted))


def mape(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    MAPE - Mean Absolute Percentage Error

    This is an arithmetic average of absolute value of percentage error

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Mean Absolute Percentage Error
    """

    return np.mean(np.abs(_percentage_error(actual, predicted)))


def smape(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    SMAPE - Symmetric Mean Absolute Percentage Error

    This is symmetric version of mean absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Symmetric Mean Absolute Percentage Error
    """

    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def mdape(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    MDAPE - Median Absolute Percentage Error

    Median value of absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Median Absolute Percentage Error
    """

    return np.median(np.abs(_percentage_error(actual, predicted)))


def smdape(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    SMDAPE - Symmetric Median Absolute Percentage Error

    This is a symmetric version of median absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Symmetric Median Absolute Percentage Error
    """

    return np.median(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def maape(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    MAAPE - Mean Arctangent Absolute Percentage Error

    This is mean of arctangent of absolute percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Mean Arctangent Absolute Percentage Error
    """

    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def rmspe(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    RMSPE - Root Mean Squared Percentage Error

    This is a square root from the mean of the percentage error.

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Root Mean Squared Percentage Error
    """

    return np.sqrt(np.mean(_percentage_error(actual, predicted)))


def rmdspe(
    actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]
):
    """
    RMDSPE - Root Median Squared Percentage Error

    This is square root from the meadian value of squared percentage error

    Note: The result is not multiplied by 100

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Root Median Squared Percentage Error
    """

    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def nape(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    NAPE - Normalized Absolute Percentage Error

    This is normalized version of absolute percentage error

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Normaized Absolute Percentage Error
    """

    __mape = mape(actual, predicted)

    return np.sqrt(
        np.sum(np.square(_percentage_error(actual, predicted) - __mape))
        / (len(actual) - 1)
    )


# Relative Errors


def mre(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    benchmark: Union[np.ndarray, pd.Series],
):
    """
    MRE - Mean Relative Error

    This is mean of relative error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values
        benchmark (Union[np.ndarray, pd.Series]): array of benchmark values

    Returns:
        Mean Relative Error
    """

    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    RAE - Relative Absolute Error

    A.K.A - Approximation Error

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Relative Absolute Error
    """

    return np.sum(np.abs(actual - predicted)) / (
        np.sum(np.abs(actual - np.mean(actual))) + EPSILON
    )


def mrae(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    benchmark: Union[np.ndarray, pd.Series],
):
    """
    MRAE - Mean Relative Absolute Error

    This is the mean of relative absolute error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values
        benchmark (Union[np.ndarray, pd.Series]): array of benchmark values

    Returns:
        Mean Relative Absolute Error
    """

    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    benchmark: Union[np.ndarray, pd.Series],
):
    """
    MDRAE - Median Relative Absolute Error

    This is median of relative absolute error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values
        benchmark (Union[np.ndarray, pd.Series]): array of benchmark values

    Returns:
        Median Relative Absolute Error
    """

    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def rrse(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    RRSE - Root Relative squared Error

    This is square root from relative squared error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Root Relative Squared Error
    """

    return np.sqrt(
        np.sum(np.square(actual - predicted))
        / np.sum(np.square(actual - np.mean(actual)))
    )


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """
    GMRAE - Geometric Mean Relative Absolute Error

    This is the geometric mean of relative absolute error.

    Args:
        actual (np.ndarray): array of actual values
        predicted (np.ndarray): array of predicted values
        benchmark (np.ndarray): array of benchmark values

    Returns:
        Geometric Mean Relative Absolute Error
    """

    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    benchmark: Union[np.ndarray, pd.Series] = None,
):
    """
    MBRAE - Mean Bounded Relative Absolute Error

    This is mean of bounded relative absolute error.

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values
        benchmark (Union[np.ndarray, pd.Series]): array of benchmark values

    Returns:
        Mean Bounded Relative Absolute Error
    """

    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    benchmark: Union[np.ndarray, pd.Series] = None,
):
    """
    UMBRAE - Unscaled Mean Bounded Relative Absolute Error

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values
        benchmark (Union[np.ndarray, pd.Series]): array of benchmark values

    Returns:
        Unscaled Mean Bounded Relative Absolute Error
    """

    __mbrae = mbrae(actual, predicted, benchmark)

    return __mbrae / (1 - __mbrae)


def mda(actual: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]):
    """
    MDA - Mean Directional Accuracy

    Args:
        actual (Union[np.ndarray, pd.Series]): array of actual values
        predicted (Union[np.ndarray, pd.Series]): array of predicted values

    Returns:
        Mean Directional Accuracy
    """

    return np.mean(
        (
            np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])
        ).astype(int)
    )
