import pathlib

import pytest

from tests import FixtureRequest
from tlab_analysis import typing


@pytest.fixture(params=["str", "Path"])
def filepath(request: FixtureRequest[str], filename: str) -> typing.FilePath:
    match request.param:
        case "str":
            return str(filename)
        case "Path":
            return pathlib.Path(filename)
    raise NotImplementedError  # pragma: no cover
