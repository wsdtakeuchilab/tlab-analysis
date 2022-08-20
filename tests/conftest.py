import pathlib

import pytest

from tests import FixtureRequest
from tlab_analysis import typing


@pytest.fixture(params=["str", "Path"])
def filepath(request: FixtureRequest[str], filepathstr: str) -> typing.FilePath:
    match request.param:
        case "str":
            return str(filepathstr)
        case "Path":
            return pathlib.Path(filepathstr)
    raise NotImplementedError  # pragma: no cover
