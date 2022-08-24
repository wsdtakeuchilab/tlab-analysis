import typing as t

import pandas as pd


def find_scdc(  # SCDC: the Start Coordinates of a Decay Curve
    xdata: t.Sequence[float | int],
    ydata: t.Sequence[float | int],
    _window: int = 10,
    _k: int = 2,
) -> tuple[float, float]:
    """
    Find the start coordinates of a decay curve.

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
    xdata: t.Sequence[float | int],
    ydata: t.Sequence[float | int],
    _alpha: float = 0.10,
) -> tuple[float, float]:
    """
    Determine a range of x axis for fitting to a decay curve.

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
    left = df["x"][df["y"].shift(2).argmax()]
    right = df["x"][
        (df.index > df["y"].argmax()) & (df["y"].ge(alpha * df["y"].max()))
    ].max()
    return float(left), float(right)
