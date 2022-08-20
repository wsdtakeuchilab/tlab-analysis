import abc
import functools

import pandas as pd


class AbstractData(abc.ABC):
    """
    Abstract class for Data.
    """

    @abc.abstractmethod
    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """
        A dataframe.
        """
