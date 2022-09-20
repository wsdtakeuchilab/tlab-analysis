#! /usr/bin/env bash

set -e
set -x

FILES_TO_FORMAT="src tests scripts"
black $FILES_TO_FORMAT
isort $FILES_TO_FORMAT
