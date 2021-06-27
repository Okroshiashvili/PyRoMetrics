import numpy as np
import pytest


@pytest.fixture
def data():

    actual = np.array([544, 422, 269, 241, 424, 376, 464, 572, 254, 296])
    predicted = np.array([529, 474, 301, 325, 346, 381, 461, 544, 280, 310])

    return actual, predicted
