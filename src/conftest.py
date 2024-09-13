import numpy as np
import pandas as pd
import pytest

from tlab_analysis import trpl


@pytest.fixture()
def trpl_data() -> trpl.TRPLData:
    """
    For tlab_analysis.trpl.TRPLData.
    """
    np.random.seed(0)
    time = np.linspace(0, 10, 3, dtype=np.float32)
    wavelength = np.linspace(200, 400, 3, dtype=np.float32)
    intensity = np.random.randint(0, 100, len(time) * len(wavelength), dtype=np.uint16)
    data = trpl.TRPLData(
        pd.DataFrame(
            dict(
                time=np.repeat(time, len(wavelength)),
                wavelength=np.tile(wavelength, len(time)),
                intensity=intensity,
            )
        )
    )
    return data
