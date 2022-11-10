#! /usr/bin/env bash

set -euC
set -o pipefail
set -x

FILES_TO_FORMAT=("src" "tests" "scripts")

black "${FILES_TO_FORMAT[@]}"
isort "${FILES_TO_FORMAT[@]}"
