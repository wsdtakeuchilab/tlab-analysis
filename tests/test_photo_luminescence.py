import io
import os
import typing as t
from unittest import mock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tests import FixtureRequest
from tlab_analysis import photo_luminescence as pl
from tlab_analysis import typing

WAVELENGTH_RESOLUTION = 640
TIME_RESOLUTION = 480


@pytest.fixture()
def header() -> bytes:
    return bytes.fromhex(
        "49 4d cd 01 80 02 e0 01 00 00 00 00 02 00 00 00"
        "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
        "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
        "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
    )


@pytest.fixture()
def metadata() -> str:
    metadata = (
        "HiPic,1.0,100,1.0,0,0,4,8,0,0,0,01-01-1970,00:00:00,"
        "0,0,0,0,0, , , , ,0,0,0,0,0, , ,0,, , , ,0,0,, ,0,0,0,0,0,0,0,0,0,0,"
        "2,1,nm,*0614925,2,1,ns,*0619021,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0.0,0,0,"
        "StopCondition:PhotonCounting, Frame=10000, Time=673.1[sec], CountingRate=0.13[%]\n"
        "Streak:Time=10 ns, Mode=Operate, Shutter=0, MCPGain=12, MCPSwitch=1, \n"
        "Spectrograph:Wavelength=490.000[nm], Grating=2 : 150g/mm, SlitWidthIn=100[um], Mode=Spectrograph\n"
        "Date:2022/01/01,00:00:00\n"
    )
    return metadata


@pytest.fixture()
def streak_image() -> npt.NDArray[np.uint16]:
    random = np.random.RandomState(0)
    return random.randint(
        0, 64, WAVELENGTH_RESOLUTION * TIME_RESOLUTION, dtype=np.uint16
    )


@pytest.fixture()
def wavelength() -> npt.NDArray[np.float32]:
    return np.linspace(435, 535, WAVELENGTH_RESOLUTION, dtype=np.float32)


@pytest.fixture()
def time() -> npt.NDArray[np.float32]:
    return np.linspace(0, 10, TIME_RESOLUTION, dtype=np.float32)


@pytest.fixture()
def data(
    header: bytes,
    metadata: str,
    streak_image: npt.NDArray[np.uint16],
    wavelength: npt.NDArray[np.float32],
    time: npt.NDArray[np.float32],
) -> pl.Data:
    return pl.Data(
        header=header,
        metadata=metadata,
        time=time,
        wavelength=wavelength,
        intensity=streak_image.astype(np.float32),
    )


@pytest.fixture()
def raw(
    header: bytes,
    metadata: str,
    streak_image: npt.NDArray[np.uint16],
    wavelength: npt.NDArray[np.float32],
    time: npt.NDArray[np.float32],
) -> bytes:
    return (
        header
        + metadata.encode("UTF-8")
        + streak_image.astype(np.uint16).tobytes("C")
        + wavelength.tobytes("C").ljust(1024 * 4, b"\x00")
        + time.tobytes("C").ljust(1024 * 4, b"\x00")
    )


@pytest.fixture()
def write_raw(filepath: typing.FilePath, raw: bytes, tmpdir: str) -> None:
    os.chdir(tmpdir)
    with open(filepath, "wb") as f:
        f.write(raw)


@pytest.mark.parametrize("filename", ["photo_luminescence_testcase.img"])
@pytest.mark.usefixtures(write_raw.__name__)
def test_read_img(data: pl.Data, filepath: typing.FilePath) -> None:
    assert pl.read_img(filepath) == data


def test_read_img_from_buffer(data: pl.Data, raw: bytes) -> None:
    with io.BytesIO(raw) as f:
        assert pl.read_img(f) == data


def test_load_img_invalid_type() -> None:
    with pytest.raises(ValueError):
        pl.read_img(None)  # type: ignore


def describe_data() -> None:
    def test_df(data: pl.Data) -> None:
        df = pd.DataFrame(
            dict(
                time=np.repeat(data.time, len(data.wavelength)),
                wavelength=np.tile(data.wavelength, len(data.time)),
                intensity=data.intensity,
            )
        )
        pd.testing.assert_frame_equal(data.df, df)

    def test_streak_image(data: pl.Data) -> None:
        streak_image = data.intensity.reshape(len(data.time), len(data.wavelength))
        np.testing.assert_equal(data.streak_image, streak_image)

    def test_resolve_along_time(data: pl.Data) -> None:
        tr = data.resolve_along_time()
        assert tr.data == data

    @pytest.mark.parametrize("time_range", [None, (0.0, 1.0)])
    def test_resolve_along_time_time_range(
        data: pl.Data, time_range: tuple[float, float] | None
    ) -> None:
        if time_range is None:
            time_range = data.time.min(), data.time.max()
        tr = data.resolve_along_time(time_range)
        assert tr.range == time_range

    def test_resolve_along_wavelength(data: pl.Data) -> None:
        wr = data.resolve_along_wavelength()
        assert wr.data == data

    @pytest.fixture()
    def find_scdc_mock() -> t.Generator[mock.Mock, None, None]:
        with mock.patch("tlab_analysis.utils.find_scdc", return_value=(1.0, 1.0)) as m:
            yield m

    @pytest.mark.usefixtures(find_scdc_mock.__name__)
    @pytest.mark.parametrize("wavelength_range", [None, (0.0, 1.0)])
    def test_resolve_along_wavelength_wavelength_range(
        data: pl.Data,
        wavelength_range: tuple[float, float] | None,
    ) -> None:
        if wavelength_range is None:
            wavelength_range = (data.wavelength.min(), data.wavelength.max())
        wr = data.resolve_along_wavelength(wavelength_range)
        assert wr.range == wavelength_range

    @pytest.mark.parametrize("time_offset", ["auto", 1.0])
    def test_resolve_along_wavelength_time_offset(
        data: pl.Data,
        time_offset: t.Literal["auto"] | float,
        find_scdc_mock: mock.Mock,
    ) -> None:
        wr = data.resolve_along_wavelength(time_offset=time_offset)
        if time_offset == "auto":
            time_offset = find_scdc_mock.return_value[0]
        assert wr.time_offset == time_offset

    @pytest.mark.parametrize("intensity_offset", ["auto", 1.0])
    def test_resolve_along_wavelength_intensity_offset(
        data: pl.Data,
        intensity_offset: t.Literal["auto"] | float,
        find_scdc_mock: mock.Mock,
    ) -> None:
        wr = data.resolve_along_wavelength(intensity_offset=intensity_offset)
        if intensity_offset == "auto":
            intensity_offset = find_scdc_mock.return_value[1]
        assert wr.intensity_offset == intensity_offset

    def test_to_raw_binary(data: pl.Data, raw: bytes) -> None:
        assert data.to_raw_binary() == raw


def describe_time_resolved() -> None:
    @pytest.fixture(params=["min_max", "first_half", "out_of_range"])
    def time_range(request: FixtureRequest[str], data: pl.Data) -> tuple[float, float]:
        min, max = data.time.min(), data.time.max()
        half = (max - min) / 2
        match request.param:
            case "min_max":
                return min, max
            case "first_half":
                return min, half
            case "out_of_range":
                return min - half / 2, half / 2
        raise NotImplementedError  # pragma: no cover

    @pytest.fixture()
    def tr(data: pl.Data, time_range: tuple[float, float]) -> pl.TimeResolved:
        return pl.TimeResolved(data, time_range)

    def test_df(tr: pl.TimeResolved) -> None:
        df = (
            tr.data.df[tr.data.df["time"].between(*tr.range)]
            .groupby("wavelength")
            .sum()
            .drop("time", axis=1)
            .reset_index()
        )
        pd.testing.assert_frame_equal(tr.df, df)

    def test_peak_wavelength(tr: pl.TimeResolved) -> None:
        peak_wavelength = float(tr.df["wavelength"][tr.smoothed_intensity().argmax()])
        assert tr.peak_wavelength == peak_wavelength

    def test_peak_intensity(tr: pl.TimeResolved) -> None:
        peak_intensity = float(tr.smoothed_intensity().max())
        assert tr.peak_intensity == peak_intensity

    def test_half_range(tr: pl.TimeResolved) -> None:
        intensity = tr.smoothed_intensity()
        wavelength = tr.df["wavelength"]
        under_half = intensity < intensity.max() / 2
        left = wavelength[
            (wavelength < wavelength[intensity.argmax()]) & under_half
        ].max()
        right = wavelength[
            (wavelength > wavelength[intensity.argmax()]) & under_half
        ].min()
        half_range = (
            float(left if left is not np.nan else wavelength.min()),
            float(right if right is not np.nan else wavelength.max()),
        )
        assert tr.half_range == half_range

    def test_FWHM(tr: pl.TimeResolved) -> None:
        left, right = tr.half_range
        FWHM = abs(right - left)
        assert tr.FWHM == FWHM

    @pytest.mark.parametrize("window", [3, 5])
    def test_smoothed_intensity(tr: pl.TimeResolved, window: int) -> None:
        smoothed_intensity = tr.df.rolling(window, center=True).mean()["intensity"]
        pd.testing.assert_series_equal(
            tr.smoothed_intensity(window), smoothed_intensity
        )


def describe_wavelength_resolved() -> None:
    @pytest.fixture(params=["min_max", "first_half", "out_of_range"])
    def wavelength_range(
        request: FixtureRequest[str], data: pl.Data
    ) -> tuple[float, float]:
        min, max = data.wavelength.min(), data.wavelength.max()
        half = (max - min) / 2
        match request.param:
            case "min_max":
                return min, max
            case "first_half":
                return min, half
            case "out_of_range":
                return min - half / 2, half / 2
        raise NotImplementedError  # pragma: no cover

    @pytest.fixture()
    def time_offset() -> float:
        return pl.WavelengthResolved.time_offset

    @pytest.fixture()
    def intensity_offset() -> float:
        return pl.WavelengthResolved.intensity_offset

    @pytest.fixture()
    def wr(
        data: pl.Data,
        wavelength_range: tuple[float, float],
        time_offset: float,
        intensity_offset: float,
    ) -> pl.WavelengthResolved:
        return pl.WavelengthResolved(
            data, wavelength_range, time_offset, intensity_offset
        )

    @pytest.mark.parametrize("has_applied_offset", [True, False])
    def test_has_applied_offset(
        wr: pl.WavelengthResolved, has_applied_offset: bool
    ) -> None:
        object.__setattr__(wr, "_has_applied_offset", has_applied_offset)
        assert wr.has_applied_offset == has_applied_offset

    @pytest.mark.parametrize("time_offset", [0.0, 1.0])
    @pytest.mark.parametrize("intensity_offset", [0.0, 1.0])
    def test_apply_offset(
        wr: pl.WavelengthResolved, time_offset: float, intensity_offset: float
    ) -> None:
        df = wr.df.copy()
        df["time"] -= time_offset
        df["intensity"] -= intensity_offset
        object.__setattr__(wr, "_has_applied_offset", False)
        wr.apply_offset()
        pd.testing.assert_frame_equal(wr.df, df)

    def test_apply_offset_alreay_applied(wr: pl.WavelengthResolved) -> None:
        object.__setattr__(wr, "_has_applied_offset", True)
        with pytest.raises(ValueError):
            wr.apply_offset()

    @pytest.mark.parametrize("time_offset", [0.0, 1.0])
    @pytest.mark.parametrize("intensity_offset", [0.0, 1.0])
    def test_unapply_offset(
        wr: pl.WavelengthResolved, time_offset: float, intensity_offset: float
    ) -> None:
        df = wr.df.copy()
        df["time"] += time_offset
        df["intensity"] += intensity_offset
        object.__setattr__(wr, "_has_applied_offset", True)
        wr.unapply_offset()
        pd.testing.assert_frame_equal(wr.df, df)

    def test_unapply_offset_not_applied_yet(wr: pl.WavelengthResolved) -> None:
        object.__setattr__(wr, "_has_applied_offset", False)
        with pytest.raises(ValueError):
            wr.unapply_offset()

    @pytest.mark.parametrize("time_offset", [0.0, 1.0])
    @pytest.mark.parametrize("intensity_offset", [0.0, 1.0])
    def test_df(wr: pl.WavelengthResolved) -> None:
        df = (
            wr.data.df[wr.data.df["wavelength"].between(*wr.range)]
            .groupby("time")
            .sum()
            .drop("wavelength", axis=1)
            .reset_index()
        )
        df["time"] -= wr.time_offset
        df["intensity"] -= wr.intensity_offset
        pd.testing.assert_frame_equal(wr.df, df)
