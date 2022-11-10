#! /usr/bin/env bash

set -euC
set -o pipefail
set -x

FILES_TO_FORMAT=("src" "tests")

flake8 "${FILES_TO_FORMAT[@]}"
black --check "${FILES_TO_FORMAT[@]}"
isort --check-only "${FILES_TO_FORMAT[@]}"
mypy "${FILES_TO_FORMAT[@]}"
