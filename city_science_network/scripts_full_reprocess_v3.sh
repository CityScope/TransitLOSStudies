#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

CITIES=(beerseba boston concepcion gipuzkoa guadalajara hamburg san_francisco shanghai taipei toronto)

for c in "${CITIES[@]}"; do
  rm -rf "$c/streets" "$c/results"
  ts=$(date +%Y%m%d_%H%M%S)
  log="$c/logs/full_reprocess_${ts}.log"
  mkdir -p "$c/logs"
  echo "=== [$(date)] starting $c ===" | tee -a "$log"
  uv run python "$c/run.py" --rebuild-tiles --use-pmtiles >> "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [$(date)] $c DONE (exit 0) ===" | tee -a "$log"
  else
    echo "=== [$(date)] $c FAILED (exit $rc) -- see $log ===" | tee -a "$log"
  fi
done
echo "=== ALL CITIES PROCESSED ==="
