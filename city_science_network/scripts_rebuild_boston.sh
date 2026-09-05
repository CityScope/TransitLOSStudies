#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Wait for scripts_relaunch_remaining.sh to finish first (avoid two
# uv/rayon tile-build processes competing for the same CPU budget).
while pgrep -f "scripts_relaunch_remaining.sh" > /dev/null; do
  sleep 15
done

ts=$(date +%Y%m%d_%H%M%S)
log="boston/logs/rebuild_map_only_${ts}.log"
mkdir -p boston/logs
echo "=== [$(date)] starting boston ===" | tee -a "$log"
uv run python boston/run.py --map-only --rebuild-tiles --use-pmtiles >> "$log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "=== [$(date)] boston DONE (exit 0) ===" | tee -a "$log"
else
  echo "=== [$(date)] boston FAILED (exit $rc) -- see $log ===" | tee -a "$log"
fi
