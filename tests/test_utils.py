from collections import abc

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from tlab_analysis import utils


@pytest.mark.parametrize(
    "invalid_xdata_and_ydata",
    [([0.0], [0.0, 0.0]), ([], [])],
    ids=["length_not_equal", "empty"],
)
def test_validate_xdata_and_ydata_invalid(
    invalid_xdata_and_ydata: tuple[abc.Sequence[float], abc.Sequence[float]]
) -> None:
    xdata, ydata = invalid_xdata_and_ydata
    with pytest.raises(ValueError):
        utils.validate_xdata_and_ydata(xdata, ydata)


@pytest.mark.parametrize("window", [3, 5, 7])
def test_smooth_when_window_is_int(window: int) -> None:
    x = list(range(100))
    smoothed = utils.smooth(x, window)
    assert (
        smoothed
        == pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_list()
    )


@pytest.mark.parametrize("window", [0.3, 0.5, 0.7])
def test_smooth_when_window_is_float(window: float) -> None:
    x = list(range(100))
    smoothed = utils.smooth(x, window)
    _window = int(len(x) * window)
    assert (
        smoothed
        == pd.Series(x).rolling(_window, center=True, min_periods=1).mean().to_list()
    )


@pytest.mark.parametrize("window", [-0.3, -3])
def test_smooth_when_window_is_negative(window: int | float) -> None:
    x = list(range(100))
    with pytest.raises(ValueError):
        utils.smooth(x, window)


def describe_peak() -> None:
    @pytest.mark.parametrize(
        "peak",
        [
            utils.Peak(x=0, y=1, x0=-0.5, x1=0.5, y0=0.5),
            utils.Peak(x=0, y=1, x0=-0.2, x1=0.2, y0=0.8),
            utils.Peak(x=0.2, y=1, x0=-0.2, x1=0.5, y0=0.5),
        ],
    )
    def test_width(peak: utils.Peak) -> None:
        assert peak.width == peak.x1 - peak.x0


@pytest.mark.parametrize("mu", [-0.5, 0.1, 0.5])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("y_max", [1.0, 3.0, 5.0])
def test_find_peaks(
    mu: float,
    sigma: float,
    y_max: float,
) -> None:
    x = np.linspace(-1.0, 1.0, 200)
    y = y_max * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    peaks = utils.find_peaks(x.tolist(), y.tolist())
    assert len(peaks) == 1, f"too many peaks found: {len(peaks)}"
    # For peak
    x_width = x.max() - x.min()
    y_width = y.max() - y.min()
    x_err = abs((peaks[0].x - mu) / x_width)
    y_err = abs((peaks[0].y - y_max) / y_width)
    eps = 1e-2
    assert x_err < eps, f"x_err is too large: {x_err:.6g}"
    assert y_err < eps, f"y_err is too large: {y_err:.6g}"
    # For peak width
    HWHM = np.sqrt(2 * np.log(2)) * sigma
    x0_acc = max(-HWHM + mu, x.min())
    x1_acc = min(HWHM + mu, x.max())
    x0_err = abs((peaks[0].x0 - x0_acc) / x_width)
    x1_err = abs((peaks[0].x1 - x1_acc) / x_width)
    assert x0_err < eps, f"x0_err is too large: {x0_err:.6g}"
    assert x1_err < eps, f"x1_err is too large: {x1_err:.6g}"


@pytest.mark.parametrize("x0", [0.1, 0.2])
@pytest.mark.parametrize("y0", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("seed", [0, 1])
def test_find_scdc(x0: float, y0: float, tau: float, seed: int) -> None:
    x = np.linspace(0, 1, 500)
    np.random.seed(seed)
    noise = np.random.normal(y0, 1e-2, size=x.size)
    y = np.where(x - x0 > 0, np.exp(-(x - x0) / tau), 0) + noise
    scdc = utils.find_scdc(list(x), list(y))
    x_err = abs((scdc[0] - x0) / (x.max() - x.min()))
    y_err = abs((scdc[1] - y0) / y.max())
    eps = 1e-2
    assert x_err < eps, f"x_err is too large: {x_err:.6g}"
    assert y_err < eps, f"y_err is too large: {y_err:.6g}"


@pytest.mark.parametrize("x0", [0.1, 0.2])
@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("seed", [0, 1])
def test_determine_fit_range_dc(x0: float, tau: float, seed: int) -> None:
    x = np.linspace(0, 1, 500)
    np.random.seed(seed)
    noise = np.random.normal(0, 1e-2, size=x.size)
    y = np.where(x - x0 > 0, np.exp(-(x - x0) / tau), 0) + noise
    fit_range = utils.determine_fit_range_dc(list(x), list(y))
    index = (x > fit_range[0]) & (x < fit_range[1])
    params, _ = optimize.curve_fit(
        lambda x, a, tau: a * np.exp(-(x - x0) / tau),
        x[index],
        y[index],
        bounds=(0, np.inf),
    )
    a_err = abs((params[0] - 1.0) / (y.max() - y.min()))
    tau_err = abs((params[1] - tau) / (x.max() - x.min()))
    eps = 1e-2
    assert a_err < eps, f"a_err is too large: {a_err:.6g}, {params[0]:.6g}"
    assert tau_err < eps, f"tau_err is too large: {tau_err:.6g}"


@pytest.mark.parametrize("a", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("x0", [-0.5, -0.25, 0.0, 0.25, 0.5])
@pytest.mark.parametrize("seed", [0, 1])
def test_curve_fit(a: float, x0: float, seed: int) -> None:
    x0 = x0 * np.pi
    x = np.linspace(0, 2 * np.pi, 100)
    np.random.seed(seed)
    noise = np.random.uniform(-a * 0.1, a * 0.1, size=x.size)
    y = a * np.sin(x - x0) + noise
    params, cov = utils.curve_fit(
        lambda x, a, x0: a * np.sin(x - x0),
        list(x),
        list(y),
        bounds=[(0, -np.inf), (np.inf, np.inf)],
    )
    a_err = abs((params[0] - a) / (2 * a))
    x0_err = abs((params[1] - x0) / (x.max() - x.min()))
    eps = 1e-2
    assert a_err < eps, f"a_err is too large: {a_err:.6g},  {a:.6g}, {params[0]:.6g}"
    assert x0_err < eps, f"x0_err is too large: {x0_err:.6g}, {x0:.6g}, {params[1]:.6g}"
