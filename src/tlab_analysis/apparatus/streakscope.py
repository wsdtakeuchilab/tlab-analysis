import copy

import numpy as np

from tlab_analysis import trpl


def correct(data: trpl.TRPLData) -> trpl.TRPLData:
    """
    Corrects wavlength and intensity of the data.

    Parameters
    ----------
    data : tlab_analysis.trpl.TRPLData
        The data to correct.

    Returns
    -------
    tlab_analysis.trpl.TRPLData
        A new TRPLData object of the corrected data.

    Examples
    --------
    >>> data = getfixture("trpl_data")
    >>> data.wavelength[:3]
    0    200.0
    1    300.0
    2    400.0
    Name: wavelength, dtype: float64
    >>> corrected = correct(data)
    >>> corrected.wavelength[:3]
    0     800.0
    1    1200.0
    2    1600.0
    Name: wavelength, dtype: float64
    >>> data.intensity[:3]
    0    44.0
    1    47.0
    2    68.0
    Name: intensity, dtype: float64
    >>> corrected.intensity[:3]
    0     0.029833
    1     1.857857
    2    68.000000
    Name: intensity, dtype: float64

    The returned object is different from the original.
    >>> data is corrected
    False
    """
    corrected_data = copy.deepcopy(data)
    corrected_data.df["wavelength"] *= 4
    corrected_data.df["intensity"] *= corrected_data.df["wavelength"].apply(
        _correct_coefficient
    )
    return corrected_data


def _correct_coefficient(wavelength: float) -> float:
    if (800 <= wavelength) & (wavelength < 900):
        c = 1 / (20637 * np.exp(-0.0033342 * wavelength) + 41.9243465)
    elif (900 <= wavelength) & (wavelength < 955):
        c = 1 / (72006 * np.exp(-0.0047107 * wavelength) + 30.7619935)
    elif (955 <= wavelength) & (wavelength < 1016):
        c = 1 / (1.0609 * 1e7 * np.exp(-0.0099372 * wavelength) + 29.6390678)
    elif (1016 <= wavelength) & (wavelength < 1600):
        c = 1 / (4.5899 * 1e9 * np.exp(-0.015847 * wavelength))
    else:
        c = 1.0
    return float(c)
