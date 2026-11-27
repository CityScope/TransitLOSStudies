#!/bin/bash
# Everything FIT_RESULTS.md reports: shared fits (pools and variants) with rankings, each fitted city on its own, the reach table, the report.
cd "$(dirname "$0")/.."
PY=${TORCH_PYTHON:-$HOME/.venvs/torch/bin/python}
bash flex/run_fits.sh large_khmax15 large_khmax1 large four five large_kh6 large_hus large_hus_wus
for c in boston san_francisco toronto concepcion guadalajara; do $PY flex/fit_cities.py own $c; echo "own $c exit $? $(date +%T)"; done
$PY flex/fit_cities.py reach; echo "reach exit $?"
$PY flex/fit_cities.py report; echo "report exit $?"
echo "all done $(date +%F\ %T)"
