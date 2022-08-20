import typing as t

import pytest

PT = t.TypeVar("PT")


class FixtureRequest(pytest.FixtureRequest, t.Generic[PT]):
    param: PT
