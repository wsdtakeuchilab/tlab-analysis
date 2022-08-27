import os
import pathlib

import pytest

from tests import FixtureRequest
from tlab_analysis import typing


@pytest.fixture(params=["str", "Path"])
def filepath(
    request: FixtureRequest[str], filename: str, tmpdir: str
) -> typing.FilePath:
    match request.param:
        case "str":
            return os.path.join(tmpdir, str(filename))
        case "Path":
            return pathlib.Path(tmpdir) / filename
        case _:
            raise NotImplementedError
