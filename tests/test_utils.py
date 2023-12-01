from collections import abc

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from tests import FixtureRequest
from tlab_analysis import utils


@pytest.fixture(params=["length_not_equal", "empty"])
def invalid_xdata_and_ydata(
    request: FixtureRequest[str],
) -> tuple[abc.Sequence[float], abc.Sequence[float]]:
    match request.param:
        case "length_not_equal":
            return [0.0], [0.0, 0.0]
        case "empty":
            return [], []
        case _:
            raise NotImplementedError


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


@pytest.mark.skip("deprecated on version 0.5.0")
@pytest.mark.parametrize("mu", [-0.5, 0.1, 0.5])
@pytest.mark.parametrize("y_max", [1.0, 3.0, 5.0])
def test_find_peak(mu: float, y_max: float) -> None:
    x = np.linspace(-1.0, 1.0, 200)
    y = y_max * np.exp(-((x - mu) ** 2))
    x_peak, y_peak = utils.find_peak(x.tolist(), y.tolist())
    x_err = abs((x_peak - mu) / (x.max() - x.min()))
    y_err = abs((y_peak - y_max) / (y.max() - y.min()))
    eps = 1e-2
    assert x_err < eps, f"x_err is too large: {x_err:.6g}"
    assert y_err < eps, f"y_err is too large: {y_err:.6g}"


@pytest.mark.skip("deprecated on version 0.5.0")
@pytest.mark.parametrize("mu", [0.0, 0.5])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.3])
def test_find_half_range(mu: float, sigma: float) -> None:
    x = np.linspace(-1.0, 1.0, 200)
    y = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    left, right = utils.find_half_range(x.tolist(), y.tolist())
    HWHM = np.sqrt(2 * np.log(2)) * sigma
    left_acc = max(-HWHM, x.min() - mu) + mu
    right_acc = min(HWHM, x.max() - mu) + mu
    left_err = abs((left - left_acc) / (x.max() - x.min()))
    right_err = abs((right - right_acc) / (x.max() - x.min()))
    eps = 1e-2
    assert left_err < eps, f"left_err is too large: {left_err:.6g}"
    assert right_err < eps, f"right_err is too large: {right_err:.6g}"


@pytest.mark.skip("deprecated on version 0.5.0")
@pytest.mark.parametrize("mu", [0.0, 0.5])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.3])
def test_find_FWHM(mu: float, sigma: float) -> None:
    x = np.linspace(-1.0, 1.0, 200)
    y = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    FWHM = utils.find_FWHM(x.tolist(), y.tolist())
    left, right = utils.find_half_range(x.tolist(), y.tolist())
    assert FWHM == abs(right - left)


@pytest.mark.parametrize("x0", [0.1, 0.2])
@pytest.mark.parametrize("y0", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("seed", [0, 1])
def test_find_scdc(x0: float, y0: float, tau: float, seed: int) -> None:
    x = np.linspace(0, 1, 500)
    np.random.seed(seed)
    noise = np.random.normal(y0, 1e-2, size=len(x))
    y = np.where(x - x0 > 0, np.exp(-(x - x0) / tau), 0) + noise
    scdc = utils.find_scdc(list(x), list(y))
    x_err = abs((scdc[0] - x0) / (x.max() - x.min()))
    y_err = abs((scdc[1] - y0) / (y.max() - y.min()))
    eps = 1e-2
    assert x_err < eps, f"x_err is too large: {x_err:.6g}"
    assert y_err < eps, f"y_err is too large: {y_err:.6g}"


def test_find_scdc_ValueError(
    invalid_xdata_and_ydata: tuple[abc.Sequence[float], abc.Sequence[float]]
) -> None:
    xdata, ydata = invalid_xdata_and_ydata
    with pytest.raises(ValueError):
        utils.find_scdc(xdata, ydata)


@pytest.mark.parametrize("x0", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("seed", [0, 1])
def test_determine_fit_range_dc(x0: float, tau: float, seed: int) -> None:
    x = np.linspace(0, 1, 500)
    np.random.seed(seed)
    noise = np.random.normal(0, 1e-2, size=len(x))
    y = np.where(x - x0 > 0, np.exp(-(x - x0) / tau), 0) + noise
    fit_range = utils.determine_fit_range_dc(list(x), list(y))
    index = (x > fit_range[0]) & (x < fit_range[1])
    params, _ = optimize.curve_fit(
        lambda x, a, tau: a * np.exp(-(x - x0) / tau), x[index], y[index]
    )
    a_err = abs(params[0] - 1.0) / (y.max() - y.min())
    tau_err = abs(params[1] - tau) / (x.max() - x.min())
    eps = 1e-2
    assert a_err < eps, f"a_err is too large: {a_err:.6g}"
    assert tau_err < eps, f"tau_err is too large: {tau_err:.6g}"


def test_determine_fit_range_dc_ValueError(
    invalid_xdata_and_ydata: tuple[abc.Sequence[float], abc.Sequence[float]]
) -> None:
    xdata, ydata = invalid_xdata_and_ydata
    with pytest.raises(ValueError):
        utils.determine_fit_range_dc(xdata, ydata)
