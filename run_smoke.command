#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
if ! .venv/bin/python -c 'import yaml, openai, requests, bs4, dateutil, numpy, faiss' >/dev/null 2>&1; then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
fi
.venv/bin/python -B scripts/smoke_one.py "$@"
