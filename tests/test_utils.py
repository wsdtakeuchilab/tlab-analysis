import typing as t

import numpy as np
import pytest
from scipy import optimize

from tests import FixtureRequest
from tlab_analysis import utils


@pytest.fixture(params=["length_not_equal", "empty"])
def invalid_xdata_and_ydata(
    request: FixtureRequest[str],
) -> tuple[t.Sequence[float], t.Sequence[float]]:
    match request.param:
        case "length_not_equal":
            return [0.0], [0.0, 0.0]
        case "empty":
            return [], []
    raise NotImplementedError  # pragma: no cover


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
    invalid_xdata_and_ydata: tuple[t.Sequence[float], t.Sequence[float]]
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
    invalid_xdata_and_ydata: tuple[t.Sequence[float], t.Sequence[float]]
) -> None:
    xdata, ydata = invalid_xdata_and_ydata
    with pytest.raises(ValueError):
        utils.determine_fit_range_dc(xdata, ydata)
