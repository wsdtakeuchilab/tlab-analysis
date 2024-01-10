from __future__ import annotations

import bisect
import dataclasses
import typing as t
from collections import abc

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, signal


def validate_xdata_and_ydata(
    xdata: abc.Collection[float], ydata: abc.Collection[float]
) -> None:
    """
    Validates `xdata` and `ydata`.

    Parameters
    ----------
    xdata : collections.abc.Collection[float]
        The independent data for x axis.
    ydata : collections.abc.Collection[float]
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


def smooth(x: abc.Collection[float], window: int | float = 3) -> list[float]:
    """
    Smooths an array by mean filtering.

    Parameters
    ----------
    x : collections.abc.Collection[float]
        A sequence of numbers to be smoothed.
    window : int | float
        The window size for mean filtering.
        if `window` < 1, the value will be interpreted
        as the ratio to the length of `x`.

    Returns
    -------
    list[float]
        A list of smoothed values.

    Examples
    --------
    >>> x = np.arange(10)
    >>> smooth(x)
    [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5]
    >>> smooth(x, window=3)
    [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5]
    >>> smooth(x, window=0.3)
    [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5]
    """
    if window < 0:
        raise ValueError(f"`window` must be a positive number: {window}")
    elif 0 < window <= 1:
        _window = int(len(x) * window)
    else:
        _window = int(window)
    return (
        pd.Series(list(x)).rolling(_window, center=True, min_periods=1).mean().to_list()
    )


@dataclasses.dataclass(frozen=True)
class Peak:
    x: float
    """The horizontal value of the peak point."""
    y: float
    """The vertical value of the peak point."""
    x0: float = dataclasses.field(repr=False)
    """The left end of the peak witdh."""
    x1: float = dataclasses.field(repr=False)
    """The right end of the peak width."""
    y0: float = dataclasses.field(repr=False)
    """The peak width height."""

    @property
    def width(self) -> float:
        """The full width at half maximum."""
        return self.x1 - self.x0


def find_peaks(
    xdata: abc.Collection[float],
    ydata: abc.Collection[float],
    *,
    spline_size: int = 1000,
    width: int = 50,
    width_height_ratio: float = 0.5,
    **kwargs: t.Any,
) -> list[Peak]:
    """
    Finds peaks of a curve.

    Parameters
    ----------
    xdata : collections.abc.Collection[float]
        The independent data for x axis.
    ydata : collections.abc.Collection[float]
        The dependent data for y axis.
    spline_size : int
        The number of points for spline interpolation.
    width : int
        Required width of peaks in samples.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        for detail.
    width_height_ratio : float
        The ratio of peak width height to peak height.
    kwargs : Any
        The keyword arguments for `scipy.signal.find_peaks`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        for detail.

    Returns
    -------
    list[Peak]
        A list of Peak object found in the curve.

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
    >>> find_peaks(x, y)
    [Peak(x=-0.002002002002001957, y=0.9999676167328615)]
    """
    # Create a B-spline curve
    spl = interpolate.make_smoothing_spline(xdata, ydata)
    x = np.linspace(min(xdata), max(xdata), spline_size)
    y = spl(x)
    # Extract peaks from the B-spline curve
    peaks, props = signal.find_peaks(y, width=width, **kwargs)
    # Create a list of Peak objects
    results: list[Peak] = list()
    for peak in peaks:
        # Finds peak half
        _spl = interpolate.UnivariateSpline(x, y - y[peak] * width_height_ratio, s=0)
        _roots = _spl.roots()
        _idx = bisect.bisect_left(_roots, x[peak])
        results.append(
            Peak(
                x=x[peak],
                y=y[peak],
                x0=_roots[_idx - 1] if _idx > 0 else x[0],
                x1=_roots[_idx] if _idx < len(_roots) else x[-1],
                y0=y[peak] * width_height_ratio,
            )
        )
    return results


def find_decay_range(
    x: abc.Collection[float],
    high: float = 0.9,
    low: float = 0.1,
) -> tuple[int, int]:
    """
    Finds the range of decay.

    Parameters
    ----------
    x : collections.abc.Collection[float]
        A sequence of numbers to find the decay range.
    high : float
        The ratio of the upper end to the maximum of `x`.
    low : float
        The ratio of the lower end to the maximum of `x`.

    Returns
    -------
    tuple[int, int]
        The indices of the decay range.

    Examples
    --------
    >>> x = [0, 0, 4, 12, 9, 6, 4, 3, 2, 2, 1, 0, 0]
    >>> find_decay_range(x)
    (4, 10)
    >>> x[4:10]
    [9, 6, 4, 3, 2, 2]
    >>> find_decay_range(x, high=0.7)
    (5, 10)
    >>> find_decay_range(x, low=0.2)
    (4, 8)
    """
    _x = np.array(x)
    max_index = _x.argmax()
    max_x = _x.max()
    left = (
        bisect.bisect_left(np.array(_x[max_index:] <= max_x * high), True) + max_index
    )
    right = (
        bisect.bisect_left(np.array(_x[max_index:] <= max_x * low), True) + max_index
    )
    return int(left), int(right)


def find_scdc(  # SCDC: the Start Coordinates of a Decay Curve
    xdata: abc.Collection[float],
    ydata: abc.Collection[float],
    *,
    _window: int = 10,
    _k: int = 2,
) -> tuple[float, float]:
    """
    Finds the start coordinates of a decay curve.

    Parameters
    ----------
    xdata : collections.abc.Collection[float]
        The independent data for x axis.
    ydata : collections.abc.Collection[float]
        The dependent data for y axis.

    Returns
    -------
    tuple[float, float]
        Coordinates of a start point of rising curve: (x, y).

    Examples
    --------
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
    validate_xdata_and_ydata(xdata, ydata)
    # Determine a range of the background signal
    window, k = _window, _k
    df = pd.DataFrame(dict(x=xdata, y=ydata))
    rolling = df["y"].rolling(window)
    sup_noise = rolling.mean().to_numpy() + k * rolling.std().to_numpy()
    background = df["y"][
        df["x"].le(df["x"][df["y"].gt(sup_noise).shift(-1, fill_value=False)].min())
    ]
    baseline = background[
        background.between(background.quantile(0.05), background.quantile(0.95))
    ].mean()
    # Determine (x, y) coordinates of a start point
    index = np.arange(len(xdata))[
        (df.index < df["y"].argmax()) & (df["y"] < baseline)
    ].max()
    return (float(df["x"][index]), float(df["y"][index]))


def determine_fit_range_dc(
    xdata: abc.Collection[float],
    ydata: abc.Collection[float],
    *,
    spline_size: int = 1000,
    _decay_ratio: float = 0.10,
) -> tuple[float, float]:  # pragma: no cover
    """
    Determines a range of x axis for fitting to a decay curve.

    Parameters
    ----------
    xdata : collections.abc.Collection[float]
        The independent data for x axis.
    ydata : collections.abc.Collection[float]
        The dependent data for y axis.
    spline_size : int
        The number of points for spline interpolation.
    _decay_ratio : float
        ToDo: Add description.

    Returns
    -------
    tuple[float, float]
        A range of x.

    Examples
    --------
    >>> x = np.linspace(-5, 5, 1000)
    >>> x0 = -2
    >>> y = np.where(x - x0 > 0, np.exp(- (x - x0)), 0)
    >>> determine_fit_range_dc(x, y)
    (-1.9369369369369371, 0.3053053053053052)

    See Also
    --------
    https://github.com/wasedatakeuchilab/tlab-analysis/blob/master/resources/images/utils/determine_fit_range_dc.svg
    """
    import warnings

    warnings.warn(
        f"{determine_fit_range_dc.__name__} is deprecated "
        "and will be removed after version 0.6.0. "
        f"Use {find_decay_range.__name__} instead.",
        category=FutureWarning,
        stacklevel=2,
    )
    validate_xdata_and_ydata(xdata, ydata)
    # Create a B-Spline curve
    spl = interpolate.make_smoothing_spline(xdata, ydata)
    _x = np.linspace(min(xdata), max(xdata), spline_size)
    _y = spl(_x)
    # Find the fitting range
    df = pd.DataFrame(dict(x=_x, y=_y))
    left = df["x"][int(df["y"].shift(2).argmax())]
    right = df["x"][
        (df.index > df["y"].argmax()) & (df["y"].ge(_decay_ratio * df["y"].max()))
    ].max()
    return float(left), float(right)


def curve_fit(
    func: t.Any,
    xdata: abc.Collection[float],
    ydata: abc.Collection[float],
    *,
    spline_size: int = 1000,
    **kwargs: t.Any,
) -> t.Any:
    """
    Fits a non-linear function to data.

    Parameters
    ----------
    func : Any
        The model function, f(x, ...).
    xdata : collections.abc.Collection[float]
        The independent data for x axis.
    ydata : collections.abc.Collection[float]
        The dependent data for y axis.
    spline_size : int
        The number of points for spline interpolation.
    kwargs : Any
        The keyword arguments for `scipy.optimize.curve_fit`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        for detail.

    Returns
    -------
    Any
        The same as what `scipy.optimize.curve_fit` returns.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        for detail.

    Examples
    --------
    >>> a = 1.0
    >>> x0 = np.pi / 4
    >>> x = np.linspace(0, 2 * np.pi, 100)
    >>> noise = np.random.uniform(-a * 0.1, a * 0.1, size=x.size)
    >>> y = a * np.sin(x - x0) + noise
    >>> f = lambda x, a, x0: a * np.sin(x - x0)
    >>> params, cov = curve_fit(f, x, y)
    >>> params
    array([1.00244564, 0.7667709 ])

    See also
    --------
    https://github.com/wasedatakeuchilab/tlab-analysis/blob/master/resources/images/utils/curve_fit.svg
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html.
    """
    validate_xdata_and_ydata(xdata, ydata)
    # Create a B-Spline curve
    spl = interpolate.make_smoothing_spline(xdata, ydata)
    _x = np.linspace(min(xdata), max(xdata), spline_size)
    _y = spl(_x)
    return optimize.curve_fit(
        func,
        xdata=_x,
        ydata=_y,
        **kwargs,
    )
