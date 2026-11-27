#!/bin/bash
# The reported shared fits and rankings, in order (each: fit with held-out scores, then the ranking). Needs TORCH_PYTHON (torch + transitlos).
cd "$(dirname "$0")/.."
PY=${TORCH_PYTHON:-$HOME/.venvs/torch/bin/python}
echo "chain started $(date +%F\ %T)"
for v in "$@"; do
  pool=${v%%_*}
  $PY flex/fit_cities.py shared $pool $v; echo "shared $v exit $? $(date +%T)"
  $PY flex/fit_cities.py rank $v; echo "rank $v exit $? $(date +%T)"
done
echo "chain done $(date +%F\ %T)"
