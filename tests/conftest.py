import os
import pathlib

import pytest

from tests import FixtureRequest


@pytest.fixture(params=["str", "Path"])
def filepath(
    request: FixtureRequest[str], filename: str, tmpdir: str
) -> str | os.PathLike[str]:
    match request.param:
        case "str":
            return os.path.join(tmpdir, str(filename))
        case "Path":
            return pathlib.Path(tmpdir) / filename
        case _:
            raise NotImplementedError


@pytest.fixture()
def write_raw_binary(filepath: os.PathLike[str], raw_binary: bytes) -> None:
    with open(filepath, "wb") as f:
        f.write(raw_binary)
