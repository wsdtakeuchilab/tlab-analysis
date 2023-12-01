import typing as t
import warnings
from collections import abc

import numpy as np
import pandas as pd

NT = t.TypeVar("NT", int, float)


def validate_xdata_and_ydata(xdata: abc.Sequence[NT], ydata: abc.Sequence[NT]) -> None:
    """
    Validates `xdata` and `ydata`.

    Parameters
    ----------
    xdata : collections.abc.Sequence[NT@validate_xdata_and_ydata]
        The independent data for x axis.
    ydata : collections.abc.Sequence[NT@validate_xdata_and_ydata]
        The dependent data for y axis.

    Raises
    ------
    ValueError
        If either `xdata` or `ydata` is invalid.

    Notes
    -----
    Validation List
        - `xdata` and `ydata` must be the same length.
        - Both `xdata` and `ydata` must not be empty.
    """
    if not len(xdata) == len(ydata):
        raise ValueError("The length of `xdata` and `ydata` must be the same")
    if len(xdata) == 0 or len(ydata) == 0:
        raise ValueError("`xdata` and `ydata` must not be empty")


def smooth(x: abc.Sequence[NT], window: int = 3) -> list[float]:
    """
    Smooths an array by mean filtering.

    Parameters
    ----------
    x : collections.abc.Sequence[NT@smooth]
        A sequence of numbers to be smoothed.
        The window size for mean filtering.
    window : int
        The window size for mean filtering.

    Returns
    -------
    list[NT@smooth]
        A list of smoothed values.

    Examples
    --------
    >>> x = np.arange(10)
    >>> smooth(x)
    [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5]
    """
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_list()


def find_peak(
    xdata: abc.Sequence[NT], ydata: abc.Sequence[NT]
) -> tuple[NT, NT]:  # pragma: no cover
    """
    Finds the peak point.

    Parameters
    ----------
    xdata : collections.abc.Sequence[NT@find_peak]
        The independent data for x axis.
    ydata : collections.abc.Sequence[NT@find_peak]
        The dependent data for y axis.

    Returns
    -------
    tuple[NT@find_peak, NT@find_peak]
        The (x, y) value at peak.

    Examples
    --------
    >>> x = np.linspace(-2, 2, 11)
    >>> x
    array([-2. , -1.6, -1.2, -0.8, -0.4,  0. ,  0.4,  0.8,  1.2,  1.6,  2. ])
    >>> y = np.exp(-x**2)
    >>> y
    array([0.01831564, 0.07730474, 0.23692776, 0.52729242, 0.85214379,
           1.        , 0.85214379, 0.52729242, 0.23692776, 0.07730474,
           0.01831564])
    >>> find_peak(x, y)
    (0.0, 1.0)
    """
    warnings.warn(
        f"{find_peak.__name__} is deprecated and will be removed after version 0.5.0. "
        f"Use scipy.signal.find_peaks instead.",
        stacklevel=2,
    )
    validate_xdata_and_ydata(xdata, ydata)
    peak_index = np.array(ydata).argmax()
    return list(xdata)[peak_index], list(ydata)[peak_index]


def find_half_range(
    xdata: abc.Sequence[NT], ydata: abc.Sequence[NT]
) -> tuple[NT, NT]:  # pragma: no cover
    """
    Finds the two x values at which y is half maximum.

    Parameters
    ----------
    xdata : collections.abc.Sequence[NT@find_half_range]
        The independent data for x axis.
    ydata : collections.abc.Sequence[NT@find_half_range]
        The dependent data for y axis.

    Returns
    -------
    tuple[NT@find_half_range, NT@find_half_range]
        The two x values on left and right side of peak.

    Examples
    --------
    >>> x = np.linspace(-2, 2, 11)
    >>> x
    array([-2. , -1.6, -1.2, -0.8, -0.4,  0. ,  0.4,  0.8,  1.2,  1.6,  2. ])
    >>> y = np.exp(-x**2)
    >>> y
    array([0.01831564, 0.07730474, 0.23692776, 0.52729242, 0.85214379,
           1.        , 0.85214379, 0.52729242, 0.23692776, 0.07730474,
           0.01831564])
    >>> find_half_range(x, y)
    (-1.2, 1.2000000000000002)
    """
    warnings.warn(
        f"{find_half_range.__name__} is deprecated and will be removed after version 0.5.0. "
        f"Use scipy.signal.find_peaks instead.",
        stacklevel=2,
    )
    validate_xdata_and_ydata(xdata, ydata)
    xarray = np.array(xdata)
    yarray = np.array(ydata)
    under_half = yarray < yarray.max() / 2
    left = xarray[(xarray < xarray[yarray.argmax()]) & under_half].max()
    right = xarray[(xarray > xarray[yarray.argmax()]) & under_half].min()
    return (
        left if left is not np.nan else xarray.min(),
        right if right is not np.nan else xarray.max(),
    )


def find_FWHM(
    xdata: abc.Sequence[NT], ydata: abc.Sequence[NT]
) -> NT:  # pragma: no cover
    """
    Finds the full width at half maximum..

    Parameters
    ----------
    xdata : collections.abc.Sequence[NT@find_FWHM]
        The independent data for x axis.
    ydata : collections.abc.Sequence[NT@find_FWHM]
        The dependent data for y axis.

    Returns
    -------
    NT@find_FWHM
        The value of FWHM.

    Examples
    --------
    >>> x = np.linspace(-2, 2, 11)
    >>> x
    array([-2. , -1.6, -1.2, -0.8, -0.4,  0. ,  0.4,  0.8,  1.2,  1.6,  2. ])
    >>> y = np.exp(-x**2)
    >>> y
    array([0.01831564, 0.07730474, 0.23692776, 0.52729242, 0.85214379,
           1.        , 0.85214379, 0.52729242, 0.23692776, 0.07730474,
           0.01831564])
    >>> find_FWHM(x, y)
    2.4000000000000004
    """
    warnings.warn(
        f"{find_FWHM.__name__} is deprecated and will be removed after version 0.5.0. "
        f"Use scipy.signal.find_peaks instead.",
        stacklevel=2,
    )
    validate_xdata_and_ydata(xdata, ydata)
    left, right = find_half_range(xdata, ydata)
    return abs(right - left)


def find_scdc(  # SCDC: the Start Coordinates of a Decay Curve
    xdata: abc.Sequence[float | int],
    ydata: abc.Sequence[float | int],
    _window: int = 10,
    _k: int = 2,
) -> tuple[float, float]:
    """
    Finds the start coordinates of a decay curve.

    Parameters
    ----------
    xdata : Sequence[float]
        The independent data for x axis.
    ydata : Sequence[float]
        The dependent data for y axis.

    Returns
    -------
    tuple[float, float]
        Coordinates of a start point of rising curve: (x, y).

    Raises
    ------
    ValueError
        If the length of `xdata` and `ydata` is not the same or zero.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 1000)
    >>> np.random.seed(222)
    >>> x0, y0 = -2, 0.1
    >>> noise = np.random.normal(y0, 1e-3, size=len(x))
    >>> y = np.where(x - x0 > 0, np.exp(- (x - x0)), 0) + noise
    >>> find_scdc(x, y)
    (-2.017017017017017, 0.09983161220321969)

    See Also
    --------
    https://github.com/wasedatakeuchilab/tlab-analysis/blob/master/resources/images/utils/find_scdc.svg
    """
    if not len(xdata) == len(ydata) != 0:
        raise ValueError(
            "The length of `xdata` and `ydata` must be the same and non-zero."
        )
    window, k = _window, _k
    df = pd.DataFrame(dict(x=xdata, y=ydata)).sort_values(by="x", ignore_index=True)
    # Determine a range of the background signal
    rolling = df["y"].rolling(window)
    sup_noise = rolling.mean().to_numpy() + k * rolling.std().to_numpy()
    background = df["y"][
        df["x"].le(df["x"][df["y"].gt(sup_noise).shift(-1, fill_value=False)].min())
    ]
    baseline = background[
        background.between(background.quantile(0.05), background.quantile(0.95))
    ].mean()
    # Determine (x, y) coordinates of a start point
    index = df.index.to_numpy()[
        (df.index < df["y"].argmax()) & (df["y"].lt(baseline))
    ].max()
    return (float(df["x"][index]), float(df["y"][index]))


def determine_fit_range_dc(
    xdata: abc.Sequence[float | int],
    ydata: abc.Sequence[float | int],
    _alpha: float = 0.10,
) -> tuple[float, float]:
    """
    Determines a range of x axis for fitting to a decay curve.

    Parameters
    ----------
    xdata : Sequence[float]
        The independent data for x axis.
    ydata : Sequence[float]
        The dependent data for y axis.

    Raises
    ------
    ValueError
        If the length of `xdata` and `ydata` is not the same or zero.

    Returns
    -------
    tuple[float, float]
        A range of x.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 1000)
    >>> x0 = -2
    >>> y = np.where(x - x0 > 0, np.exp(- (x - x0)), 0)
    >>> determine_fit_range_dc(x, y)
    (-1.9769769769769772, 0.3053053053053052)

    See Also
    --------
    https://github.com/wasedatakeuchilab/tlab-analysis/blob/master/resources/images/utils/determine_fit_range_dc.svg
    """
    if not len(xdata) == len(ydata) != 0:
        raise ValueError(
            "The length of `xdata` and `ydata` must be the same and non-zero."
        )
    alpha = _alpha
    df = pd.DataFrame(dict(x=xdata, y=ydata)).sort_values(by="x", ignore_index=True)
    left = df["x"][int(df["y"].shift(2).argmax())]
    right = df["x"][
        (df.index > df["y"].argmax()) & (df["y"].ge(alpha * df["y"].max()))
    ].max()
    return float(left), float(right)
