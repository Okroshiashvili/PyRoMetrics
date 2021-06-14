import numpy as np

EPSILON = 1e-10


def _error(actual, predicted):

    return actual - predicted


def _percentage_error(actual, predicted):

    return _error(actual, predicted) / (actual + EPSILON)


def mse(actual, predicted):

    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual, predicted):

    return np.sqrt(mse(actual, predicted))


def me(actual, predicted):

    return np.mean(_error(actual, predicted))


def mae(actual, predicted):

    return np.mean(np.abs(_error(actual, predicted)))


def mpe(actual, predicted):

    return np.mean(_percentage_error(actual, predicted))


def mape(actual, predicted):

    return np.mean(np.abs(_percentage_error(actual, predicted)))
