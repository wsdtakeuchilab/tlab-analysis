from __future__ import annotations

import dataclasses
import functools
import io
import os
import typing as t
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize

from tlab_analysis import abstract, typing, utils

DEFAULT_HEADER = bytes.fromhex(
    "49 4d cd 01 80 02 e0 01 00 00 00 00 02 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
)
DEFAULT_METADATA = (
    "HiPic,1.0,100,1.0,0,0,4,8,0,0,0,01-01-1970,00:00:00,"
    "0,0,0,0,0, , , , ,0,0,0,0,0, , ,0,, , , ,0,0,, ,0,0,0,0,0,0,0,0,0,0,2,"
    "1,nm,*0614925,2,1,ns,*0619021,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0.0,0,0,"
    "StopCondition:PhotonCounting, Frame=10000, Time=300.0[sec], CountingRate=0.10[%]\n"
    "Streak:Time=10 ns, Mode=Operate, Shutter=0, MCPGain=10, MCPSwitch=1,\n"
    "Spectrograph:Wavelength=490.000[nm], Grating=2 : 150g/mm, SlitWidthIn=100[um], Mode=Spectrograph\n"
    "Date:1970/01/01,00:00:00\n"
)


def read_img(filepath_or_buffer: typing.FilePathOrBuffer) -> Data:
    """
    Read and parse a raw binary file generated by u8167 application.

    Parameters
    ----------
    filepath_or_buffer : FilePathOrBuffer
        The path to a raw binary or buffer from u8167.

    Returns
    -------
    tlab_analysis.photo_luminescence.Data
        A Data object from the raw file.

    Raises
    ------
    ValueError
        If `filepath_or_buffer` is invalid.

    Examples
    --------
    >>> data = load_img("data.img")  # doctest: +SKIP
    """
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        with open(filepath_or_buffer, "rb") as f:
            return _read_img(f)
    elif isinstance(filepath_or_buffer, io.BufferedIOBase):
        return _read_img(filepath_or_buffer)
    else:
        raise ValueError(
            "The type of filepath_or_buffer must be FilePath or io.BufferedIOBase"
        )


def _read_img(file: io.BufferedIOBase) -> Data:
    sector_size = 1024
    wavelength_resolution = 640
    time_resolution = 480
    header = file.read(64)
    metadata = [file.readline() for _ in range(4)]
    intensity = np.frombuffer(file.read(sector_size * 600), dtype=np.uint16)
    wavelength = np.frombuffer(file.read(sector_size * 4), dtype=np.float32)[
        :wavelength_resolution
    ]
    time = np.frombuffer(file.read(sector_size * 4), dtype=np.float32)[:time_resolution]
    return Data(
        header=header,
        metadata=b"".join(metadata).decode("UTF-8"),
        intensity=intensity.astype(np.float32),
        wavelength=wavelength.astype(np.float32),
        time=time.astype(np.float32),
    )


@dataclasses.dataclass()
class Data(abstract.AbstractData):
    """
    Data of photo luminescence experiments.

    Examples
    --------
    >>> import numpy as np

    ### Construct directly

    Create a time array.
    >>> time_resolution = 480
    >>> time = np.linspace(
    ...     0, 10, time_resolution,
    ...     dtype=np.float32
    ... )

    Create a wavelength array.
    >>> wavelength_resolution = 640
    >>> wavelength = np.linspace(
    ...     400, 500, wavelength_resolution,
    ...     dtype=np.float32
    ... )

    Create an intensity array.
    >>> intensity = np.random.randint(
    ...     0, 32, time_resolution * wavelength_resolution
    ... ).astype(np.float32)

    Create a Data object.
    >>> data = Data(time, wavelength, intensity)
    """

    time: npt.NDArray[np.float32]
    """A 1D array of time in nanosecond."""
    wavelength: npt.NDArray[np.float32]
    """A 1D array of wavelength in nanometer."""
    intensity: npt.NDArray[np.float32]
    """A 1D array of intensity in arbitrary units."""
    header: bytes = DEFAULT_HEADER
    """Bytes of the header of a raw binary from u8167."""
    metadata: str = DEFAULT_METADATA
    """Meta information of the data from u8167."""

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Data):
            return all(
                (
                    np.all(self.time == __o.time),
                    np.all(self.wavelength == __o.wavelength),
                    np.all(self.intensity == __o.intensity),
                    self.header == __o.header,
                    self.metadata == __o.metadata,
                )
            )
        else:
            return NotImplemented  # pragma: no cover

    @functools.cached_property
    def streak_image(self) -> npt.NDArray[np.float32]:
        """
        A 2D array of a streak image.
        """
        img_size = len(self.wavelength) * len(self.time)
        intensity = np.zeros(img_size, dtype=np.float32)
        s = min(img_size, len(self.intensity))
        intensity[:s] = self.intensity[:s]
        img = intensity.reshape(len(self.time), len(self.wavelength))
        return img

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """
        A dataframe.

        Columns
        -------
        time : float
            Time in nanosecond.
        wavelength : float
            Wavelength in nanometer.
        intensity : float
            Intensity in arbitrary units.
        """
        df = pd.DataFrame(
            dict(
                time=np.repeat(self.time, len(self.wavelength)),  # [ns]
                wavelength=np.tile(self.wavelength, len(self.time)),  # [nm]
                intensity=self.streak_image.flatten(),  # [arb. units]
            )
        )
        return df

    def resolve_along_time(
        self,
        time_range: tuple[float, float] | None = None,
    ) -> "TimeResolved":
        """
        Resolve data along the time axis.

        Parameters
        ----------
        time_range : tuple[float, float] | None
            A range of time for resolution.
            If None, the whole time of data is used.

        Returns
        -------
        tlab_analysis.photo_luminescence.TimeResolved
            A time-resolved data of photo luminescence.
        """
        assert "wavelength" in self.df.columns
        assert "intensity" in self.df.columns
        if time_range is None:
            time_range = self.time.min(), self.time.max()
        return TimeResolved(self, time_range)

    def resolve_along_wavelength(
        self,
        wavelength_range: tuple[float, float] | None = None,
        time_offset: t.Literal["auto"] | float = "auto",
        intensity_offset: t.Literal["auto"] | float = "auto",
    ) -> "WavelengthResolved":
        """
        Resolve data along the wavelength axis.

        Parameters
        ----------
        wavelength_range : tuple[float, float] | None
            A range of wavelength for resolution.
            If None, the whole wavelength of data is used.
        time_offset : "auto" | float
            An offset value of time.
            `auto` will work if the intensity is a decay curve.
        intensity_offset : "auto" | float
            An offset value of intensity.
            `auto` will work if the intensity is a decay curve.

        Returns
        -------
        tlab_analysis.photo_luminescence.WavelengthResolved
            A wavelength-resolved data of photo luminescence.
        """
        assert "time" in self.df.columns
        assert "intensity" in self.df.columns
        if wavelength_range is None:
            wavelength_range = self.wavelength.min(), self.wavelength.max()
        if time_offset == "auto" or intensity_offset == "auto":
            wr = WavelengthResolved(self, wavelength_range)
            scdc = utils.find_scdc(
                wr.df["time"].to_list(), wr.df["intensity"].to_list()
            )
            time_offset = scdc[0] if time_offset == "auto" else time_offset
            intensity_offset = (
                scdc[1] if intensity_offset == "auto" else intensity_offset
            )
        return WavelengthResolved(self, wavelength_range, time_offset, intensity_offset)

    def to_raw_binary(self) -> bytes:
        """
        Convert to a raw binary that u8167 can operate.

        Returns
        -------
        bytes
            A raw binary that u8167 can operate.
        """
        sector_size = 1024
        intensity_size = sector_size * 600
        wavelength_size = time_size = sector_size * 4
        data = (
            self.header
            + self.metadata.encode("UTF-8")
            + self.intensity.astype(np.uint16)
            .tobytes("C")
            .ljust(intensity_size, b"\x00")[:intensity_size]
            + self.wavelength.astype(np.float32)
            .tobytes("C")
            .ljust(wavelength_size, b"\x00")[:wavelength_size]
            + self.time.astype(np.float32)
            .tobytes("C")
            .ljust(time_size, b"\x00")[:time_size]
        )
        return data


@dataclasses.dataclass(frozen=True)
class TimeResolved:
    """
    Time-resolved data.
    """

    data: Data
    """Data to be resolved."""
    range: tuple[float, float]
    """A range of time for resolution."""

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """
        A dataframe.

        Columns
        -------
        wavelength : float
            Wavelength in nanometer.
        intensity : float
            Intensity in arbitrary units.
        """
        assert "time" in self.data.df.columns
        assert "wavelength" in self.data.df.columns
        assert "intensity" in self.data.df.columns
        df = (
            self.data.df[self.data.df["time"].between(*self.range)]
            .groupby("wavelength")
            .sum()
            .drop("time", axis=1)
            .reset_index()
        )
        return df

    @property
    def peak_wavelength(self) -> float:
        """
        The wavelength at peak.

        It could not equal to the wavelength at the maximum intensity,
        because of smoothing before taking argmax.
        """
        assert "wavelength" in self.df.columns
        return float(self.df["wavelength"][self.smoothed_intensity().argmax()])

    @property
    def peak_intensity(self) -> float:
        """
        The intensity at peak.

        It could not equal to the maximum intensity,
        because of smoothing before taking argmax.
        """
        return float(self.smoothed_intensity().max())

    @property
    def half_range(self) -> tuple[float, float]:
        """
        The two wavelengths at which the intensity is half maximum.
        """
        assert "wavelength" in self.df.columns
        intensity = self.smoothed_intensity()
        wavelength = self.df["wavelength"]
        under_half = intensity < intensity.max() / 2
        left = wavelength[
            (wavelength < wavelength[intensity.argmax()]) & under_half
        ].max()
        right = wavelength[
            (wavelength > wavelength[intensity.argmax()]) & under_half
        ].min()
        return (
            float(left if left is not np.nan else wavelength.min()),
            float(right if right is not np.nan else wavelength.max()),
        )

    @property
    def FWHM(self) -> float:
        """
        Full width at half maximum.
        """
        left, right = self.half_range
        return abs(right - left)

    def smoothed_intensity(self, window: int = 3) -> pd.Series[t.Any]:
        """
        Return a smoothed intensity by mean filtering.

        Parameters
        ----------
        window : int
            The window size for mean filtering.

        Returns
        -------
        pandas.Series
            A Series object of the smoothed intensity.
        """
        assert "intensity" in self.df.columns
        intensity = self.df["intensity"].rolling(window, center=True).mean()
        assert isinstance(intensity, pd.Series)
        return intensity


@dataclasses.dataclass(frozen=True)
class WavelengthResolved:
    """
    Wavelength-resolved data.
    """

    data: Data
    """Data to be resolved."""
    range: tuple[float, float]
    """A range of wavelength for resolution."""
    time_offset: float = 0.0
    """An offset value of time."""
    intensity_offset: float = 0.0
    """An offset value of intensity."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_has_applied_offset", False)
        self.apply_offset()

    @property
    def has_applied_offset(self) -> bool:
        """Whether the offset values has been applied to its dataframe."""
        return bool(self.__getattribute__("_has_applied_offset"))

    def apply_offset(self) -> None:
        """
        Apply offset values to its dataframe.

        Raises
        ------
        ValueError
            If offset values have already been applied.
        """
        if not self.has_applied_offset:
            self.df["time"] -= self.time_offset
            self.df["intensity"] -= self.intensity_offset
            object.__setattr__(self, "_has_applied_offset", False)
        else:
            raise ValueError("The offset values have been already applied")

    def unapply_offset(self) -> None:
        """
        Unapply offset values to its dataframe.

        Raises
        ------
        ValueError
            If offset values have been applied yet.
        """
        if self.has_applied_offset:
            self.df["time"] += self.time_offset
            self.df["intensity"] += self.intensity_offset
            object.__setattr__(self, "_has_applied_offset", False)
        else:
            raise ValueError("The offset values are not applied yet")

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """
        A dataframe.

        Columns
        -------
        time : float
            Time in nanosecond.
        intensity : float
            Intensity in arbitrary units.
        """
        assert "time" in self.data.df.columns
        assert "wavelength" in self.data.df.columns
        assert "intensity" in self.data.df.columns
        df = (
            self.data.df[self.data.df["wavelength"].between(*self.range)]
            .groupby("time")
            .sum()
            .drop("wavelength", axis=1)
            .reset_index()
        )
        return df

    def fit(
        self,
        func: abc.Callable[[t.Any], t.Any],
        fitting_range: tuple[float, float] | None = None,
        **kwargs: t.Any,
    ) -> tuple[t.Any, t.Any]:  # pragma: no cover
        """
        Fit a non-linear function to a intensity-time curve.

        Parameters
        ----------
        func : Callable
            A function to fit.
        fitting_range : tuple[float, float] | None
            A range of time to fit.
            If None, it is determined automatically.
        **kwargs : Any
            Additional keyword arguments of `scipy.optimize.curve_fit`.

        Returns
        -------
        params : numpy.array
            Optimal values for the parameters
            so that the sum of the squared residuals of
            func(time, *params) - intensity is minimized.
        cov : 2D numpy.array
            The estimated covariance of params.

        See also
        --------
        scipy.optimize.curve_fit
        """
        assert "intensity" in self.df.columns
        assert "time" in self.df.columns
        if fitting_range is None:
            fitting_range = utils.determine_fit_range_dc(
                self.df["time"].to_list(),
                self.df["intensity"].to_list(),
            )
        df = self.df
        index = df.index[df["time"].between(*fitting_range)]
        params, cov = optimize.curve_fit(
            f=func, xdata=df["time"][index], ydata=df["intensity"][index], **kwargs
        )
        # Issue: Should these side effectes be seperated?
        df["fit"] = np.nan
        df.loc[index, "fit"] = func(df["time"][index], *params)
        return params, cov
