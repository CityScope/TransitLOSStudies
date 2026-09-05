#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Waits for the currently-running boston process (scripts_rebuild_boston_final.sh)
# to finish, then does one final full reprocess of ALL 11 cities with the
# complete area-weighted-absolute-columns fix (added after every earlier
# run in this session).
while pgrep -f "scripts_rebuild_boston_final.sh" > /dev/null; do
  sleep 30
done

CITIES=(andorra beerseba boston concepcion gipuzkoa guadalajara hamburg san_francisco shanghai taipei toronto)

for c in "${CITIES[@]}"; do
  rm -rf "$c/streets" "$c/results"
  ts=$(date +%Y%m%d_%H%M%S)
  log="$c/logs/final_pass_${ts}.log"
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
