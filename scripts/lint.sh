#! /usr/bin/env bash

set -e
set -x

FILES_TO_CHECK="src tests"

flake8 $FILES_TO_CHECK
black $FILES_TO_CHECK --check
isort $FILES_TO_CHECK --check-only
mypy $FILES_TO_CHECK
