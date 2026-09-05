#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

while pgrep -f "scripts_full_reprocess_all.sh" > /dev/null; do
  sleep 15
done

rm -rf andorra/streets andorra/results
ts=$(date +%Y%m%d_%H%M%S)
log="andorra/logs/full_reprocess_${ts}.log"
mkdir -p andorra/logs
echo "=== [$(date)] starting andorra ===" | tee -a "$log"
uv run python andorra/run.py --rebuild-tiles --use-pmtiles >> "$log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "=== [$(date)] andorra DONE (exit 0) ===" | tee -a "$log"
else
  echo "=== [$(date)] andorra FAILED (exit $rc) -- see $log ===" | tee -a "$log"
fi
