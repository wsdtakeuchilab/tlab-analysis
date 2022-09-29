#! /usr/bin/env bash

set -euC
set -o pipefail
set -x

VERSION=$(hatch version)
git add src
git commit -m "🔖 Release v${VERSION}" --allow-empty
git tag "v${VERSION}"
git push origin master "v${VERSION}"
