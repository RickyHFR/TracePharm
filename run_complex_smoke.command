#!/bin/zsh
set -eu
exec "${0:A:h}/run_smoke.command" --case complex "$@"
