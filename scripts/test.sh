#! /usr/bin/env bash

set -e
set -x

pytest --doctest-modules src
pytest --cov-report=term-missing:skip-covered ${@}
