#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# 2026-09-05: full reprocess of all 11 cities with everything fixed this
# session: crop_by_aoi_connected union-before-explode street-connectivity
# fix, h3_population.py raster-polygon-masking + 300m roadless-cell cap,
# the fast WorldPop->H3 resampler (geohierarchy.raster_resample.raster_to_h3_auto,
# wired as pycensus.worldpop_raster_to_h3_fast), the geohierarchy sum_agg
# mass-conservation bug fix (was inflating area-weighted population sums),
# the H3-resolution-coarsening density-bound clamp (_clamp_density_to_children_bounds),
# the circle-legend zoom fix (both Folium and MapLibre paths), the
# _join_polygon_stats_lightweight fix for the census-join chunk-boundary
# double-counting bug, the circle-size-by=counts-only/opacity-by=relative-
# only field split, and isochrone_chunk_h3_resolution=4 (moved from 5 on
# 2026-09-05) for the 3 large-area cities (boston/toronto/hamburg).
# Streets + results are deleted per city so nothing stale is reused --
# every stage reruns fresh.
#
# Sequential, not parallel: this host has been OOM-killed multiple times
# today running even ONE city unwrapped: every city's run is wrapped in a
# systemd-run --user --scope memory cap (20G/1G swap) as a safety net, same
# as today's successful Boston runs.

CITIES=(andorra beerseba concepcion gipuzkoa guadalajara san_francisco taipei toronto hamburg shanghai boston)

find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find ../../transitLOS ../../UrbanAccessAnalyzer ../../geohierarchy ../../pyCensus -name "__pycache__" -exec rm -rf {} + 2>/dev/null

for c in "${CITIES[@]}"; do
  rm -rf "$c/streets" "$c/results"
  ts=$(date +%Y%m%d_%H%M%S)
  log="$c/logs/final_pass_${ts}_all11.log"
  mkdir -p "$c/logs"
  echo "=== [$(date)] starting $c ===" | tee -a "$log"
  systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=1G -- \
    uv run python "$c/run.py" --rebuild-tiles --use-pmtiles --no-reuse-cached-los >> "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [$(date)] $c DONE (exit 0) ===" | tee -a "$log"
  else
    echo "=== [$(date)] $c FAILED (exit $rc) -- see $log ===" | tee -a "$log"
  fi
done
echo "=== ALL CITIES PROCESSED ==="
