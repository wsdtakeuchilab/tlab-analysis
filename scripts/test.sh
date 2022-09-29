#! /usr/bin/env bash

set -euC
set -o pipefail
set -x

# pytest --doctest-modules src  # Uncomment when to run doctests
pytest --cov-report=term-missing:skip-covered "${@}"
