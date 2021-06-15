import pandas as pd
import pytest


@pytest.fixture
def data():

    data = {
        "actual": [544, 422, 269, 241, 424, 376, 464, 572, 254, 296],
        "predicted": [529, 474, 301, 325, 346, 381, 461, 544, 280, 310],
    }

    df = pd.DataFrame(data)

    return df
