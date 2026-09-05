#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

while pgrep -f "scripts_full_reprocess_v3.sh" > /dev/null; do
  sleep 30
done

rm -rf boston/streets boston/results
ts=$(date +%Y%m%d_%H%M%S)
log="boston/logs/full_reprocess_${ts}_final.log"
mkdir -p boston/logs
echo "=== [$(date)] starting boston ===" | tee -a "$log"
uv run python boston/run.py --rebuild-tiles --use-pmtiles >> "$log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "=== [$(date)] boston DONE (exit 0) ===" | tee -a "$log"
else
  echo "=== [$(date)] boston FAILED (exit $rc) -- see $log ===" | tee -a "$log"
fi
