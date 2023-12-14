import numpy as np
import pandas as pd
import pytest

from tests import FixtureRequest
from tlab_analysis import trpl
from tlab_analysis.apparatus import streakscope

WAVELENGTH_RESOLUTION = 640
TIME_RESOLUTION = 480


@pytest.fixture(params=[0, 1, 2])
def data(request: FixtureRequest[int]) -> trpl.TRPLData:
    random = np.random.RandomState(request.param)
    time = np.linspace(0, 10, TIME_RESOLUTION, dtype=np.float32)
    wavelength = np.linspace(200, 400, WAVELENGTH_RESOLUTION, dtype=np.float32)
    df = pd.DataFrame(
        dict(
            time=np.repeat(time, len(wavelength)),
            wavelength=np.tile(wavelength, len(time)),
            intensity=random.randint(
                0, 100, WAVELENGTH_RESOLUTION * TIME_RESOLUTION, dtype=np.uint16
            ),
        )
    )
    data = trpl.TRPLData(df)
    return data


def test_correct(data: trpl.TRPLData) -> None:
    corrected = streakscope.correct(data)
    assert corrected is not data
    pd.testing.assert_series_equal(
        corrected.df["wavelength"], data.df["wavelength"] * 4
    )
