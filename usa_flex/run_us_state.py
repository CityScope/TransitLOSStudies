"""All counties of one state (run by run_us.py): candidates -> station profiles once, then every county that has a cached street network.

    python run_us_state.py VT [--force]

Reads the original study's data read-only; writes data/<State>/{candidates.parquet, stop_profiles.parquet, summary.json, <GEOID>/...}.
"""
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_county as rc  # noqa: E402
from transitlos.flex import core  # noqa: E402


def main(abbr: str, force: bool = False, cfg: core.Config = core.Config()):
    name = rc.STATE_DIRS[abbr]
    state_dir = rc.STUDY / "data" / name
    out_state = HERE / "data" / name
    out_state.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counties = sorted(p.name for p in (state_dir / "counties").iterdir() if (p / "_network_cache_b1500" / "meta.txt").is_file())
    done_marker = out_state / "summary.json"
    if done_marker.is_file() and not force:
        print(abbr, "already done"); return
    stations, prof, cut = rc.state_profiles(state_dir, out_state / "x", cfg)
    rc.write_stations(out_state, stations, prof, cut)
    t_prof = time.time() - t0
    summary = {"state": abbr, "stations": int(len(stations)), "t_profiles_s": round(t_prof, 1), "counties": {}}
    for g in counties:
        t1 = time.time()
        try:
            r = rc.run(abbr, g, cfg, shared=(stations, prof, cut), quiet=True, validate=(state_dir / "counties" / g / "h3_res10.geoparquet").is_file())
            summary["counties"][g] = {k: r[k] for k in ("cells", "cells_with_stops", "cells_capped", "cells_with_cropped", "stations_cropped_total", "max_cropped_in_a_cell",
                                                         "cells_search_cap", "t_search_s", "t_edge_cells_s", "t_cells_s", "t_recompute_s", "validation") if k in r}
            summary["counties"][g]["t_total_s"] = round(time.time() - t1, 1)
        except Exception as exc:   # one county must not stop the state
            summary["counties"][g] = {"error": repr(exc), "trace": traceback.format_exc()[-800:]}
            print(abbr, g, "FAILED", repr(exc), flush=True)
    summary["t_total_s"] = round(time.time() - t0, 1)
    done_marker.write_text(json.dumps(summary, indent=1, default=float))
    ok = sum(1 for v in summary["counties"].values() if "error" not in v)
    print(f"{abbr}: {len(stations)} stations, {ok}/{len(counties)} counties in {summary['t_total_s']}s (profiles {t_prof:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], "--force" in sys.argv)
