"""Is the analysis date the pipeline will use a good one? (`main.py --gtfs-only` runs this for every city.)

For each city (or, for `stops_per_feed` cities, each feed) it loads the GTFS exactly like the pipeline's stop-scoring step
(`transitlos.stops`), asks `pick_analysis_date` for the date, and measures how representative that date is:

  * how many trips run on it compared with the busiest / the median working day of the same window,
  * whether it is a weekday (and not a public holiday), and whether it lies inside every feed's calendar,
  * whether the feed is stale (service already ended) or the date is outside the pipeline's search horizon,
  * that at least some stops have service in the 06:00-22:00 window that day.

Writes `gtfs_date_audit.csv` next to `main.py` and returns the rows; a row's `verdict` is `OK` or says what to look at.
"""
from __future__ import annotations

import traceback
from datetime import date, time, timedelta
from pathlib import Path

import geopandas as gpd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent


def _group_audit(city: str, label: str, dirs: list[str], aoi) -> dict:
    from pyGTFSHandler.feed import Feed
    from transitlos import stops as tstops

    feed = Feed(gtfs_dirs=dirs, aoi=aoi)
    picked = tstops.pick_analysis_date(feed)
    today = date.today()
    cal_lo, cal_hi = feed.calendar.min_date, feed.calendar.max_date
    horizon = tstops._REPRESENTATIVE_DATE_HORIZON_DAYS
    lo, hi = max(cal_lo, today - timedelta(days=horizon)), min(cal_hi, today + timedelta(days=horizon))
    in_horizon = lo <= hi
    if not in_horizon:
        lo, hi = cal_lo, cal_hi
    per_service = dict(feed.trips.lf.group_by("service_id").len().collect().iter_rows())
    days = feed.calendar.get_services_in_date_range(lo, hi, date_type=None)
    note = ""
    try:
        xy = feed.stops.lf.select(pl.col("stop_lon").mean(), pl.col("stop_lat").mean()).collect().row(0)
        days = feed.calendar.filter_by_date_type(days, "workday", xy[0], xy[1])
    except Exception as exc:
        note = f"holiday lookup failed ({exc}); weekends only"
        days = feed.calendar.filter_by_date_type(days, "weekday", None, None)
    counts = {str(r[0]): sum(per_service.get(sid, 0) for sid in r[1]) for r in days.select("date", "service_ids").iter_rows()}
    vals = sorted(counts.values())
    best = vals[-1] if vals else 0
    median = vals[len(vals) // 2] if vals else 0
    on_picked = counts.get(picked.isoformat(), 0)

    served = total = 0
    try:
        feed_stops = set(feed.stops.lf.select("stop_id").collect()["stop_id"].to_list())
        total = len(feed_stops)
        hw = feed.get_headway_at_stops(date=picked, start_time=time(6), end_time=time(22))
        hw = hw.collect() if hasattr(hw, "collect") else hw
        served = len(feed_stops & set(hw["stop_id"].to_list())) if "stop_id" in hw.columns else len(hw)
    except Exception as exc:
        note = (note + "; " if note else "") + f"stop coverage failed ({type(exc).__name__}: {str(exc)[:80]})"

    problems = []
    if picked.weekday() >= 5:
        problems.append("picked date is a weekend")
    if picked.isoformat() not in counts:
        problems.append("picked date is a holiday / not a working day in the window")
    if cal_hi < today:
        problems.append(f"STALE: service ended {cal_hi}")
    if not (cal_lo <= picked <= cal_hi):
        problems.append("picked date outside the calendar")
    if best and on_picked < 0.8 * best:
        problems.append(f"only {on_picked / best:.0%} of the busiest working day's trips")
    if total and served == 0:
        problems.append("no stop has service in 06:00-22:00 on the picked date")
    if not in_horizon:
        problems.append("calendar outside the +/-horizon around today (historical feed)")
    return {
        "city": city, "group": label, "n_feeds": len(dirs), "picked_date": picked.isoformat(), "weekday": picked.strftime("%a"),
        "trips_on_date": on_picked, "busiest_workday_trips": best, "median_workday_trips": median,
        "ratio_to_busiest": round(on_picked / best, 3) if best else None,
        "calendar_min": str(cal_lo), "calendar_max": str(cal_hi), "stops_with_service_06_22": served, "stops_in_feed": total,
        "verdict": "OK" if not problems else "; ".join(problems), "note": note,
    }


def audit_city(key: str) -> list[dict]:
    from code.city_config import CITY_CONFIGS

    city_dir = ROOT / key
    gtfs = city_dir / "gtfs"
    dirs = [str(p) for p in sorted(gtfs.iterdir()) if p.is_dir()] if gtfs.is_dir() else []
    if not dirs:
        return [{"city": key, "group": "-", "verdict": "NO GTFS FOUND"}]
    aoi = None
    if (city_dir / "aoi.gpkg").is_file():
        aoi = gpd.read_file(city_dir / "aoi.gpkg").to_crs(4326)
    groups = [(Path(d).name, [d]) for d in dirs] if CITY_CONFIGS[key].stops_per_feed else [("all feeds", dirs)]
    rows = []
    for label, group in groups:
        try:
            rows.append(_group_audit(key, label, group, aoi))
        except Exception as exc:
            msg = str(exc)
            if "No stops found inside AOI" in msg or "No trips found inside your aoi" in msg:
                verdict = "SKIPPED: no service inside the city AOI (the pipeline skips this feed too)"
            elif "Cannot determine date range" in msg:
                verdict = "SKIPPED: feed has no calendar.txt / calendar_dates.txt data (unusable)"
            else:
                verdict = f"AUDIT FAILED: {type(exc).__name__}: {msg[:150]}"
            rows.append({"city": key, "group": label, "n_feeds": len(group), "verdict": verdict, "note": traceback.format_exc()[-300:] if verdict.startswith("AUDIT") else ""})
    return rows


def audit_all(keys) -> list[dict]:
    import pandas as pd

    rows = []
    for key in keys:
        print(f"[gtfs-audit] {key}", flush=True)
        for row in audit_city(key):
            print(f"   {row.get('group', ''):28.28} picked {row.get('picked_date')} ({row.get('weekday')}) trips {row.get('trips_on_date')}"
                  f" / busiest {row.get('busiest_workday_trips')} | stops with service {row.get('stops_with_service_06_22')} -> {row['verdict']}", flush=True)
            rows.append(row)
    out = ROOT / "gtfs_date_audit.csv"
    new = pd.DataFrame(rows)
    if out.is_file():  # keep the rows of cities not audited this time (--cities)
        old = pd.read_csv(out)
        new = pd.concat([old[~old["city"].isin(new["city"])], new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"[gtfs-audit] wrote {ROOT / 'gtfs_date_audit.csv'}", flush=True)
    return rows
