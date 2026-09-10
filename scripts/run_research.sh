#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  printf '%s\n' 'ERROR: project .venv is missing; run the installer first.' >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" -m counterfactual_microscopy.research "$@"
