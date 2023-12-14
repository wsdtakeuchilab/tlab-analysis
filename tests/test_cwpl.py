import io
import os

import numpy as np
import pandas as pd
import pytest

from tests import FixtureRequest
from tlab_analysis import cwpl


@pytest.fixture(params=[0, 1, 2])
def data(request: FixtureRequest[int]) -> cwpl.CWPLData:
    metadata = [
        "4300\r\n",
        "5300\r\n",
        '"                                                                "\r\n',
        "0\r\n",
        "0\r\n",
        "0\r\n",
        '"                                                                                                                                                                                                                                                                "\r\n',
        "501\r\n",
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
        '"no data"\r\n',
    ]
    random = np.random.RandomState(request.param)
    grating = np.arange(4000, 5000, step=2)
    intensity = random.random(grating.size).round(4)
    df = pd.DataFrame(
        dict(
            grating=grating,
            intensity=intensity,
        )
    )
    data = cwpl.CWPLData(df, metadata)
    return data


@pytest.fixture()
def raw_binary(data: cwpl.CWPLData) -> bytes:
    df = pd.DataFrame(
        {
            "": None,
            "x (cm)": data.grating.map(lambda x: "{0:.3f}".format(x)),
            "強度1 (mv)": data.intensity.map(lambda x: "{0:.4E}".format(x)),
            "強度2 (mV)": pd.Series(np.zeros(data.grating.size)).map(
                lambda x: "{0:.4E}".format(x)
            ),
        }
    )
    return (
        "".join(data.metadata)
        + "".join(
            f'"{line}"\r\n'
            for line in df.to_csv(
                index=False,
            ).splitlines()
        )
    ).encode(cwpl.CWPLData.HR320.encoding)


@pytest.mark.parametrize("filename", ["cwpl_testcase.img"])
@pytest.mark.usefixtures("write_raw_binary")
def test_read_file_from_str_or_pathlike(
    filepath: str | os.PathLike[str], data: cwpl.CWPLData
) -> None:
    actual = cwpl.read_file(filepath)
    assert actual == data


def test_read_file_from_buffer(raw_binary: bytes, data: cwpl.CWPLData) -> None:
    with io.BytesIO(raw_binary) as f:
        actual = cwpl.read_file(f)
    assert actual == data


def test_read_file_invalid_type() -> None:
    with pytest.raises(ValueError):
        cwpl.read_file(None)  # type: ignore


def describe_cwpl_data() -> None:
    def test_grating(data: cwpl.CWPLData) -> None:
        pd.testing.assert_series_equal(data.grating, data.df["grating"].astype(float))

    def test_wavelength(data: cwpl.CWPLData) -> None:
        pd.testing.assert_series_equal(data.wavelength, data.df["grating"] / 10)

    def test_wavelength_caribrated(data: cwpl.CWPLData) -> None:
        pd.testing.assert_series_equal(
            data.wavelength_caribrated, 1.0473 * data.wavelength - 32.273
        )

    def test_instensity(data: cwpl.CWPLData) -> None:
        pd.testing.assert_series_equal(data.intensity, data.df["intensity"])

    def test_to_raw_binary(data: cwpl.CWPLData, raw_binary: bytes) -> None:
        assert data.to_raw_binary() == raw_binary
