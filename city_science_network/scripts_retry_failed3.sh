#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

CITIES=(concepcion toronto hamburg)

find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find ../../transitLOS ../../UrbanAccessAnalyzer ../../geohierarchy ../../pyCensus -name "__pycache__" -exec rm -rf {} + 2>/dev/null

for c in "${CITIES[@]}"; do
  rm -rf "$c/streets" "$c/results"
  ts=$(date +%Y%m%d_%H%M%S)
  log="$c/logs/final_pass_${ts}_retry.log"
  mkdir -p "$c/logs"
  echo "=== [$(date)] starting $c (retry) ===" | tee -a "$log"
  systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=1G -- \
    uv run python "$c/run.py" --rebuild-tiles --use-pmtiles --no-reuse-cached-los >> "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [$(date)] $c DONE (exit 0) ===" | tee -a "$log"
  else
    echo "=== [$(date)] $c FAILED (exit $rc) -- see $log ===" | tee -a "$log"
  fi
done
echo "=== RETRY BATCH PROCESSED ==="
