import pytest


def test_mse(sample_data):

    assert all(sample_data["actual"]) == all(sample_data["predicted"])
