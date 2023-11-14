#! /usr/bin/env bash

set -euC

hatch env create
SETTING_JSON_PATH="$HOME/.vscode-server/data/Machine/settings.json"
jq '. + { "python.defaultInterpreterPath": "'"$(hatch env find)"'/bin/python"}' "$SETTING_JSON_PATH" \
    | tee "$SETTING_JSON_PATH.tmp"
mv "$SETTING_JSON_PATH.tmp" "$SETTING_JSON_PATH"
