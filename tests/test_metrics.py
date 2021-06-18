import numpy as np

from pyrometrics.metrics import mse, rmse, me, mae, mpe, mape, mdae, gmae, nrmse, smape


def test_me(data):

    ME = -8.9

    assert me(data["actual"], data["predicted"]) == ME


def test_mae(data):

    MAE = 33.7

    assert mae(data["actual"], data["predicted"]) == MAE


def test_mdae(data):

    MDAE = 27.0

    assert mdae(data["actual"], data["predicted"]) == MDAE


def test_gmae(data):

    GMAE = 21.867388

    assert np.allclose(gmae(data["actual"], data["predicted"]), GMAE)


def test_mse(data):

    MSE = 1878.3

    assert mse(data["actual"], data["predicted"]) == MSE
    assert mse(data["actual"].values, data["predicted"].values) == MSE


def test_rmse(data):

    RMSE = 43.33935

    assert np.allclose(rmse(data["actual"], data["predicted"]), RMSE)


def test_nrmse(data):

    NRMSE = 0.130934

    assert np.allclose(nrmse(data["actual"], data["predicted"]), NRMSE)


def test_mpe(data):

    MPE = -0.0486734

    assert np.allclose(mpe(data["actual"], data["predicted"]), MPE)


def test_mape(data):

    MAPE = 0.10206393

    assert np.allclose(mape(data["actual"], data["predicted"]), MAPE)


def test_smape(data):

    SMAPE = 0.096918

    assert np.allclose(smape(data["actual"], data["predicted"]), SMAPE)
