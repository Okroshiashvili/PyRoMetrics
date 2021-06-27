import numpy as np

from pyrometrics.metrics import (
    me,
    mae,
    nae,
    mase,
    mdae,
    gmae,
    mse,
    rmse,
    rmsse,
    nrmse,
    inrse,
    mpe,
    mape,
    smape,
    mdape,
    smdape,
    maape,
    rmspe,
    rmdspe,
    nape,
    mre,
    rae,
    mrae,
    mdrae,
    rrse,
    gmrae,
    mbrae,
    umbrae,
    mda,
)


def test_me(data):

    ME = -8.9

    actual, predicted = data

    assert me(actual, predicted) == ME


def test_mae(data):

    MAE = 33.7

    actual, predicted = data

    assert mae(actual, predicted) == MAE


def test_nae(data):

    NAE = 63.367096

    actual, predicted = data

    assert np.allclose(nae(actual, predicted), NAE)


def test_mase(data):

    MASE = 0.278256

    actual, predicted = data

    assert np.allclose(mase(actual, predicted), MASE)


def test_mdae(data):

    MDAE = 27.0

    actual, predicted = data

    assert mdae(actual, predicted) == MDAE


def test_gmae(data):

    GMAE = 21.867388

    actual, predicted = data

    assert np.allclose(gmae(actual, predicted), GMAE)


def test_mse(data):

    MSE = 1878.3

    actual, predicted = data

    assert mse(actual, predicted) == MSE


def test_rmse(data):

    RMSE = 43.33935

    actual, predicted = data

    assert np.allclose(rmse(actual, predicted), RMSE)


def test_rmsse(data):

    RMSSE = 0.357847

    actual, predicted = data

    assert np.allclose(rmsse(actual, predicted), RMSSE)


def test_nrmse(data):

    NRMSE = 0.130934

    actual, predicted = data

    assert np.allclose(nrmse(actual, predicted), NRMSE)


def test_inrse(data):

    INRSE = 0.381994

    actual, predicted = data

    assert np.allclose(inrse(actual, predicted), INRSE)


def test_mpe(data):

    MPE = -0.0486734

    actual, predicted = data

    assert np.allclose(mpe(actual, predicted), MPE)


def test_mape(data):

    MAPE = 0.10206393

    actual, predicted = data

    assert np.allclose(mape(actual, predicted), MAPE)


def test_smape(data):

    SMAPE = 0.096918

    actual, predicted = data

    assert np.allclose(smape(actual, predicted), SMAPE)


def test_mdape(data):

    MDAPE = 0.075656

    actual, predicted = data

    assert np.allclose(mdape(actual, predicted), MDAPE)


def test_smdape(data):

    SMDAPE = 0.073778

    actual, predicted = data

    assert np.allclose(smdape(actual, predicted), SMDAPE)


def test_maape(data):

    MAAPE = 0.1003826

    actual, predicted = data

    assert np.allclose(maape(actual, predicted), MAAPE)


def test_rmspe(data):

    RMSPE = 0.141687

    actual, predicted = data

    assert np.allclose(rmspe(actual, predicted), RMSPE)


def test_rmdspe(data):

    RMDSPE = 0.080231

    actual, predicted = data

    assert np.allclose(rmdspe(actual, predicted), RMDSPE)


def test_nape(data):

    NAPE = 0.211943

    actual, predicted = data

    assert np.allclose(nape(actual, predicted), NAPE)


def test_mre(data):

    MRE = 0.467505

    actual, predicted = data

    assert np.allclose(mre(actual, predicted, 1), MRE)


def test_rae(data):

    RAE = 0.340404

    actual, predicted = data

    np.allclose(rae(actual, predicted), RAE)


def test_mrae(data):

    MRAE = 0.541580

    actual, predicted = data

    assert np.allclose(mrae(actual, predicted, 1), MRAE)


def test_mdrae(data):

    MDRAE = 0.259259

    actual, predicted = data

    assert np.allclose(mdrae(actual, predicted, 1), MDRAE)


def test_rrse(data):

    RRSE = 0.381994

    actual, predicted = data

    assert np.allclose(rrse(actual, predicted), RRSE)


def test_gmrae(data):

    GMRAE = 0.242127

    actual, predicted = data
    assert np.allclose(gmrae(actual, predicted, 1), GMRAE)


def test_mbrae(data):

    MBRAE = 0.242160

    actual, predicted = data

    assert np.allclose(mbrae(actual, predicted, 1), MBRAE)


def test_umbrae(data):

    UMBRAE = 0.3195406

    actual, predicted = data

    assert np.allclose(umbrae(actual, predicted, 1), UMBRAE)


def test_mda(data):

    MDA = 0.777777

    actual, predicted = data

    assert np.allclose(mda(actual, predicted), MDA)
