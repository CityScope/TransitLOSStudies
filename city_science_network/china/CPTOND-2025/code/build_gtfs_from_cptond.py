"""Convert CPTOND-2025 shapefiles (bus and/or metro, one Chinese city) to GTFS.

NOTE: the Shanghai study feed is no longer built with this script -- it is
built by `build_shanghai_metro_area_gtfs.py` (whole metro area, researched
headways; see shanghai/gtfs/sources/GTFS_CONSTRUCTION.md). The Shanghai
headway tables below are the older ones, kept for the generic single-city use.

CPTOND-2025 (China Public Transport Operation Network Dataset) ships route
geometries and per-route ordered stop lists as shapefiles, not GTFS, and it
carries no timetable -- no departure times, no explicit headway. See
`TransitLOSStudies/china/sources/sources.md` for provenance.

Route polylines from `<city>_<mode>_routes.shp` are written to `shapes.txt`
(one shape per trip) so maps draw the real alignments, not stop-to-stop
straight lines.

This script builds a *frequency-based* GTFS feed (GTFS `frequencies.txt`,
exact_times=0) rather than fabricating an explicit stop-by-stop schedule:
  - one representative trip per route (route_id in the source data is
    already direction-specific, e.g. "727 Road (A--B)" is one direction),
  - in-vehicle running times between consecutive stops are estimated from
    great-circle distance at an assumed average speed (separately
    configurable for bus vs. metro), NOT measured,
  - service is assumed to run every day (no calendar info in the source)
    over the configured start/end clock window.

Headways -- what's real vs. assumed:
  - Shanghai Metro (city="shanghai", mode="metro"): headways are REAL,
    sourced from Wikipedia's "Shanghai Metro" article headway table
    (per-line morning-peak / midday / evening-peak / off-peak train
    intervals, sourced there to the operator's official site, checked
    2026-08-10), matched to each route by the line number parsed out of its
    `route_en` name. Lines with no published figure (e.g. under-construction
    extensions, the Pujiang/Jiamin shuttle lines) fall back to a generic
    placeholder (peak 5 min / midday 8 min / other 10 min) -- flagged in
    the run's stdout output as "no headway data" for those lines.
  - Bus (Shanghai specifically): no bulk per-route frequency source exists
    (checked), but real headways were found for a handful of individual
    routes/categories and are used verbatim for the routes/categories they
    describe; everything else in the same CPTOND `type_en` category gets
    that category's real-data-derived value rather than one flat guess.
    Specifically (all found 2026-08-10, see build_mode_gtfs / BUS_* below
    for the numbers and exact sources):
      * "Regular buses" (the vast majority, ~3150 of 3225 routes): based on
        Wikipedia's "Shanghai Bus Route 55" article (peak 3-5 min, off-peak
        8-12 min), cross-checked against Shanghai's own official transit
        service standard (peak headway target <=8 min central city / <=10
        min suburban) -- consistent with Route 55 being a normal, not
        unusually frequent, trunk route.
      * "DZ"-prefixed customized/demand-responsive routes (~17 routes):
        based on Wikipedia's "Shanghai Bus Route DZ418" article, which has
        a real published weekday schedule. Applied verbatim to all DZ/
        blank-`type_en` routes as the best available proxy -- flagged as
        imperfect, since further research (Shanghai city government news
        articles on the DZ program) showed other DZ routes are wildly
        different in nature (e.g. DZ301 runs once a week, DZ401/DZ416 run
        only on weekends/holidays); DZ418 is simply the only one with a
        real minute-level schedule.
      * "Tourist line" (~15 routes): based on travelchinaguide.com /
        tripadvisor-type coverage of Shanghai's sightseeing bus network
        (Puxi Essence Line ~20 min, Puxi-Pudong connector ~30 min) -- used
        flat ~20 min (sightseeing lines run flat all-day frequency, not a
        peak/off-peak pattern).
      * "Airport Limousine Bus" (2 routes): based on published Pudong
        airport shuttle-bus intervals (15-45 min band) -- used a flat ~20
        min representative value.
      * "Ferry" (~30 routes): based on Wikipedia's "Shanghai Ferry" article,
        which gives per-line intervals for a sample of named lines (10-30
        min range) -- not matched per specific CPTOND ferry route, just
        used as a category-wide informed range (peak 15 / midday 20 /
        other 25 min).
      * "Community car" (5 routes): no numeric source found anywhere
        (checked); used a flat ~20 min placeholder loosely anchored to the
        official suburban-bus headway standard, explicitly a guess, not
        data.
    All other cities' bus (and non-Shanghai metro) fall back to a flat
    generic peak/off-peak default (CLI-configurable) -- NOT measured, not
    even category-informed.

Treat this feed as a schematic reconstruction of the network's spatial
structure, adequate for network-structure / coverage analysis. Within
Shanghai, treat metro Lines 1-18 + Pujiang, and the bus category headways
above, as real/informed; treat Jiamin/Line 22 metro and "Community car" bus
frequencies as flagged guesses; treat every other city's frequencies as an
unqualified placeholder.

Usage:
    python build_gtfs_from_cptond.py \\
        --city shanghai \\
        --data-root TransitLOSStudies/CS_transitLOS/shanghai/gtfs \\
        --out TransitLOSStudies/CS_transitLOS/shanghai/gtfs/shanghai_cptond \\
        --bus-headway-min 12 --metro-headway-min 6

`--data-root` must contain `bus_shapefiles/` and/or `metro_shapefiles/`,
each with `<city>_<mode>_stops.shp` (per-route ordered stop visits) and
`<city>_<mode>_stops_unique.shp` (deduplicated stop locations), i.e. the
layout CPTOND-2025's city-organizer scripts produce.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re

import geopandas as gpd

MODE_ROUTE_TYPE = {"bus": 3, "metro": 1}  # GTFS route_type: 3=Bus, 1=Subway/Metro

# Real per-line headways (minutes), Shanghai Metro, from Wikipedia's
# "Shanghai Metro" headway table (inner-section values; peak = average of
# morning/evening peak intervals given there). Checked 2026-08-10.
SHANGHAI_METRO_HEADWAY_MIN = {
    1: {"peak": 2.75, "midday": 4.00, "other": 8.00},
    2: {"peak": 2.75, "midday": 4.00, "other": 8.00},
    3: {"peak": 5.00, "midday": 7.00, "other": 8.25},
    4: {"peak": 4.08, "midday": 8.00, "other": 8.50},
    5: {"peak": 2.75, "midday": 4.50, "other": 8.75},
    6: {"peak": 2.25, "midday": 4.50, "other": 7.50},
    7: {"peak": 2.33, "midday": 5.50, "other": 6.50},
    8: {"peak": 2.38, "midday": 4.33, "other": 5.50},
    9: {"peak": 2.17, "midday": 5.50, "other": 8.00},
    10: {"peak": 2.75, "midday": 4.00, "other": 6.50},
    11: {"peak": 2.50, "midday": 6.00, "other": 7.00},
    12: {"peak": 3.13, "midday": 6.00, "other": 8.00},
    13: {"peak": 2.75, "midday": 6.00, "other": 8.00},
    14: {"peak": 3.67, "midday": 6.00, "other": 8.50},
    15: {"peak": 4.08, "midday": 8.00, "other": 9.00},
    16: {"peak": 3.75, "midday": 7.00, "other": 7.50},
    17: {"peak": 3.75, "midday": 10.00, "other": 9.00},
    18: {"peak": 3.25, "midday": 7.00, "other": 9.25},
}
# Pujiang line: real published headway (travelchinaguide.com, checked
# 2026-08-10) -- peak = avg(07:00-09:00 4'15", 18:00-20:00 5'00"),
# midday/other = 10' (stated flat outside peak, both weekday and weekend).
PUJIANG_HEADWAY_MIN = {"peak": 4.625, "midday": 10.00, "other": 10.00}
# Jiamin Line and Line 22: NOT YET OPERATING as of 2026-08 (Jiamin Line
# expected 2027, Line 22 phase I expected end of 2026 -- confirmed via
# Wikipedia, checked 2026-08-10). CPTOND-2025 includes their planned
# alignment/stops, but there is no real operating service to source a
# headway from -- this is a guess, not an "undocumented but running" case.
GENERIC_METRO_HEADWAY_MIN = {"peak": 5.00, "midday": 8.00, "other": 10.00}


def shanghai_metro_windows(route_en, service_start, service_end):
    """3-window (peak/midday/other) frequency windows for a Shanghai metro
    route, split around a 07:00-09:30 / 17:00-19:30 peak. Returns
    (windows, matched) where windows is [(start,end,headway_min), ...] and
    matched is False if the line had no real data (generic fallback used).
    """
    route_en = route_en or ""
    if "Pujiang" in route_en:
        hw, matched = PUJIANG_HEADWAY_MIN, True
    else:
        m = re.search(r"[Ll]ine\s*(\d+)", route_en)
        line_no = int(m.group(1)) if m else None
        hw = SHANGHAI_METRO_HEADWAY_MIN.get(line_no)
        matched = hw is not None
        if hw is None:
            hw = GENERIC_METRO_HEADWAY_MIN

    bounds = sorted({service_start, "07:00", "09:30", "17:00", "19:30", service_end})
    bounds = [b for b in bounds if service_start <= b <= service_end]
    period_for = lambda s, e: "peak" if (s, e) in {("07:00", "09:30"), ("17:00", "19:30")} else (
        "midday" if (s, e) == ("09:30", "17:00") else "other")
    windows = []
    for s, e in zip(bounds, bounds[1:]):
        windows.append((s, e, hw[period_for(s, e)]))
    return windows, matched


# Shanghai bus: per CPTOND `type_en` category headway (minutes), derived
# from the real examples documented in the module docstring. Categories use
# either a peak/midday/other split or a flat all-day value, matching how
# the source data for that category actually behaves.
BUS_CATEGORY_HEADWAY_MIN = {
    "Regular buses": {"peak": 4.0, "midday": 10.0, "other": 10.0},
    "Ferry": {"peak": 15.0, "midday": 20.0, "other": 25.0},
    "Tourist line": {"flat": 20.0},
    "Airport Limousine Bus": {"flat": 20.0},
    "Community car": {"flat": 20.0},  # no source found -- explicit guess
}
BUS_DEFAULT_CATEGORY = "Regular buses"
# DZ418's real weekday schedule (Wikipedia "Shanghai Bus Route DZ418"),
# applied to all DZ-prefixed / blank-type_en customized routes as the only
# real minute-level reference available for that category (imperfect --
# other DZ routes run on very different patterns, see docstring).
DZ_WINDOWS_MIN = [
    (None, "07:30", 20.0),
    ("07:30", "09:30", 3.0),
    ("09:30", "11:30", 20.0),
    ("11:30", "13:30", 15.0),
    ("13:30", "17:30", 20.0),
    ("17:30", "19:30", 5.0),
    ("19:30", None, 20.0),
]


def peak_split_windows(hw, service_start, service_end):
    """hw: {'peak','midday','other'} or {'flat'} -> [(start,end,headway_min)]."""
    if "flat" in hw:
        return [(service_start, service_end, hw["flat"])]
    bounds = sorted({service_start, "07:00", "09:30", "17:00", "19:30", service_end})
    bounds = [b for b in bounds if service_start <= b <= service_end]
    period_for = lambda s, e: "peak" if (s, e) in {("07:00", "09:30"), ("17:00", "19:30")} else (
        "midday" if (s, e) == ("09:30", "17:00") else "other")
    return [(s, e, hw[period_for(s, e)]) for s, e in zip(bounds, bounds[1:])]


def bus_windows(route_cn, type_en, service_start, service_end):
    """Returns (windows, category_label) for one Shanghai bus route."""
    route_cn = route_cn or ""
    if route_cn.upper().startswith("DZ") or not (type_en or "").strip() or type_en == "[]":
        windows = []
        for s, e, hwmin in DZ_WINDOWS_MIN:
            s = service_start if s is None else s
            e = service_end if e is None else e
            if service_start <= s < e <= service_end:
                windows.append((s, e, hwmin))
        return windows, "DZ_custom (proxy: DZ418)"
    hw = BUS_CATEGORY_HEADWAY_MIN.get(type_en)
    if hw is None:
        return peak_split_windows(BUS_CATEGORY_HEADWAY_MIN[BUS_DEFAULT_CATEGORY], service_start, service_end), \
            f"{BUS_DEFAULT_CATEGORY} (unrecognised type_en={type_en!r})"
    return peak_split_windows(hw, service_start, service_end), type_en


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def along_line_km(line, lats, lons):
    """Cumulative distance (km) of each stop along `line`, in stop order.

    Each stop is projected onto the part of the line after the previous
    stop, so loop/out-and-back routes that pass the same spot twice still
    get monotonic positions. Segment lengths are haversine (EPSG:4326).
    """
    coords = list(line.coords)
    seg_km = [haversine_km(la1, lo1, la2, lo2) for (lo1, la1), (lo2, la2) in zip(coords, coords[1:])]
    cum_km = [0.0]
    for d in seg_km:
        cum_km.append(cum_km[-1] + d)

    def km_at(frac_idx):
        i = int(frac_idx)
        if i >= len(seg_km):
            return cum_km[-1]
        return cum_km[i] + seg_km[i] * (frac_idx - i)

    out, start = [], 0.0
    for lat, lon in zip(lats, lons):
        best = None
        for i in range(int(start), len(coords) - 1):
            (x1, y1), (x2, y2) = coords[i], coords[i + 1]
            dx, dy = x2 - x1, y2 - y1
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / seg2))
            if i == int(start):
                t = max(t, start - int(start))
            px, py = x1 + t * dx, y1 + t * dy
            d2 = (px - lon) ** 2 + (py - lat) ** 2
            if best is None or d2 < best[0] - 1e-18:
                best = (d2, i + t)
            if d2 < 1e-16:  # stop lies on this segment: take the first hit
                break
        pos = start if best is None else best[1]
        out.append(km_at(pos))
        start = pos
    return out


def hhmmss(total_seconds):
    total_seconds = int(round(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_mode(data_root: str, city: str, mode: str):
    """Return (stops_unique_gdf, per_route_stops_gdf, route_cn_to_type_en,
    route_cn_to_geometry) for one mode, or (None, None, None, None). The type_en map comes from
    <city>_<mode>_routes.shp (route_cn -> CPTOND category, e.g. "Regular
    buses", "Ferry", "Tourist line") when that file exists; routes.shp has
    no explicit route_id, so the join key is route_cn text. The geometry map
    (same key) feeds `shapes.txt`: CPTOND route lines are direction-specific
    and drawn in stop order, with every stop lying on the line (checked for
    all Shanghai bus+metro routes, 2026-09-28).
    """
    mode_dir = os.path.join(data_root, f"{mode}_shapefiles")
    stops_path = os.path.join(mode_dir, f"{city}_{mode}_stops.shp")
    unique_path = os.path.join(mode_dir, f"{city}_{mode}_stops_unique.shp")
    if not (os.path.exists(stops_path) and os.path.exists(unique_path)):
        return None, None, None, None
    stops = gpd.read_file(stops_path).to_crs(epsg=4326)
    unique = gpd.read_file(unique_path).to_crs(epsg=4326)

    type_en_by_route_cn = {}
    geom_by_route_cn = {}
    routes_path = os.path.join(mode_dir, f"{city}_{mode}_routes.shp")
    if os.path.exists(routes_path):
        routes_gdf = gpd.read_file(routes_path)
        if "route_cn" in routes_gdf.columns and "type_en" in routes_gdf.columns:
            type_en_by_route_cn = dict(zip(routes_gdf["route_cn"], routes_gdf["type_en"]))
        if "route_cn" in routes_gdf.columns:
            routes_gdf = routes_gdf.to_crs(epsg=4326)
            geom_by_route_cn = {
                rc: g for rc, g in zip(routes_gdf["route_cn"], routes_gdf.geometry)
                if g is not None and g.geom_type == "LineString"
            }

    return unique, stops, type_en_by_route_cn, geom_by_route_cn


def build_mode_gtfs(mode, city, unique_gdf, stops_gdf, type_en_by_route_cn, geom_by_route_cn, speed_kmh,
                     bus_peak_min, bus_offpeak_min,
                     service_start, service_end, service_id):
    prefix = mode[0]  # 'b' or 'm', keeps bus/metro stop/route/trip ids disjoint
    unmatched_metro_lines = 0
    bus_category_counts = {}

    stops_rows = []
    for _, row in unique_gdf.iterrows():
        stops_rows.append({
            "stop_id": f"{prefix}_{row['stop_id']}",
            "stop_name": row.get("stop_en") or row.get("stop_cn") or row["stop_id"],
            "stop_lat": row.geometry.y,
            "stop_lon": row.geometry.x,
        })

    routes_rows = []
    trips_rows = []
    stop_times_rows = []
    frequencies_rows = []
    shapes_rows = []

    stops_gdf = stops_gdf.sort_values(["route_id", "sequence"])
    for route_id, grp in stops_gdf.groupby("route_id", sort=False):
        grp = grp.sort_values("sequence")
        stop_ids = [f"{prefix}_{sid}" for sid in grp["stop_id"]]
        lats = grp.geometry.y.tolist()
        lons = grp.geometry.x.tolist()
        if len(stop_ids) < 2:
            continue

        route_cn = grp["route_cn"].iloc[0]
        route_en = grp["route_en"].iloc[0]
        routes_rows.append({
            "route_id": f"{prefix}_{route_id}",
            "route_short_name": (route_en or route_cn)[:50],
            "route_long_name": route_en or route_cn,
            "route_type": MODE_ROUTE_TYPE[mode],
        })

        # One shape per trip (each trip is one direction-specific route). A
        # route with no CPTOND line gets an empty shape_id, so map drawing
        # falls back to its stop sequence.
        geom = geom_by_route_cn.get(route_cn)
        if geom is not None:
            # Running time from distance ALONG the route line, so the speed
            # pyGTFSHandler derives from shapes.txt equals `speed_kmh`.
            dists_km = along_line_km(geom, lats, lons)
        else:
            dists_km = [0.0]
            for (lat1, lon1), (lat2, lon2) in zip(zip(lats, lons), zip(lats[1:], lons[1:])):
                dists_km.append(dists_km[-1] + haversine_km(lat1, lon1, lat2, lon2))
        offsets = [d / speed_kmh * 3600 for d in dists_km]

        trip_id = f"{prefix}_{route_id}"
        shape_id = trip_id if geom is not None else ""
        if geom is not None:
            for seq, (lon, lat) in enumerate(geom.coords):
                shapes_rows.append({
                    "shape_id": shape_id,
                    "shape_pt_lat": round(lat, 7),
                    "shape_pt_lon": round(lon, 7),
                    "shape_pt_sequence": seq,
                })
        trips_rows.append({
            "route_id": f"{prefix}_{route_id}",
            "service_id": service_id,
            "trip_id": trip_id,
            "trip_headsign": grp["route_en"].iloc[0] or grp["route_cn"].iloc[0],
            "direction_id": 0,
            "shape_id": shape_id,
        })
        for seq, (sid, off) in enumerate(zip(stop_ids, offsets)):
            t = hhmmss(off)
            stop_times_rows.append({
                "trip_id": trip_id,
                "arrival_time": t,
                "departure_time": t,
                "stop_id": sid,
                "stop_sequence": seq,
            })
        if mode == "metro" and city == "shanghai":
            windows, matched = shanghai_metro_windows(route_en, service_start, service_end)
            if not matched:
                unmatched_metro_lines += 1
        elif mode == "bus" and city == "shanghai":
            type_en = type_en_by_route_cn.get(route_cn)
            windows, category = bus_windows(route_cn, type_en, service_start, service_end)
            bus_category_counts[category] = bus_category_counts.get(category, 0) + 1
        else:
            windows = [
                (service_start, "07:00", bus_offpeak_min),
                ("07:00", "09:30", bus_peak_min),
                ("09:30", "17:00", bus_offpeak_min),
                ("17:00", "19:30", bus_peak_min),
                ("19:30", service_end, bus_offpeak_min),
            ]
            windows = [(s, e, h) for s, e, h in windows if service_start <= s < e <= service_end]

        for start, end, headway_min in windows:
            frequencies_rows.append({
                "trip_id": trip_id,
                "start_time": start + ":00",
                "end_time": end + ":00",
                "headway_secs": round(headway_min * 60),
                "exact_times": 0,
            })

    if mode == "metro" and city == "shanghai" and unmatched_metro_lines:
        print(f"  [{mode}] {unmatched_metro_lines} route(s) had no matched real headway data "
              f"(generic {GENERIC_METRO_HEADWAY_MIN} placeholder used)")
    if bus_category_counts:
        print(f"  [{mode}] category breakdown: {bus_category_counts}")

    return stops_rows, routes_rows, trips_rows, stop_times_rows, frequencies_rows, shapes_rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True, help="City pinyin key matching CPTOND filenames, e.g. 'shanghai'")
    ap.add_argument("--data-root", required=True, help="Dir containing bus_shapefiles/ and/or metro_shapefiles/")
    ap.add_argument("--out", required=True, help="Output GTFS directory (created if missing)")
    ap.add_argument("--modes", default="bus,metro", help="Comma-separated subset of: bus,metro")
    ap.add_argument("--bus-speed-kmh", type=float, default=18.0)
    ap.add_argument("--metro-speed-kmh", type=float, default=35.0)
    ap.add_argument("--bus-peak-headway-min", type=float, default=10.0,
                     help="Generic placeholder (not measured per route) for 07:00-09:30/17:00-19:30")
    ap.add_argument("--bus-offpeak-headway-min", type=float, default=18.0,
                     help="Generic placeholder (not measured per route) for the rest of the service window")
    ap.add_argument("--service-start", default="06:00")
    ap.add_argument("--service-end", default="22:00")
    ap.add_argument("--agency-name", default=None)
    args = ap.parse_args()

    speed = {"bus": args.bus_speed_kmh, "metro": args.metro_speed_kmh}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    all_stops, all_routes, all_trips, all_stop_times, all_freq, all_shapes = [], [], [], [], [], []
    found_any = False
    for mode in modes:
        unique_gdf, stops_gdf, type_en_by_route_cn, geom_by_route_cn = load_mode(args.data_root, args.city, mode)
        if unique_gdf is None:
            print(f"[skip] no {mode} shapefiles found for city={args.city!r} under {args.data_root}")
            continue
        found_any = True
        s, r, t, st, fr, sh = build_mode_gtfs(
            mode, args.city, unique_gdf, stops_gdf, type_en_by_route_cn, geom_by_route_cn, speed[mode],
            args.bus_peak_headway_min, args.bus_offpeak_headway_min,
            args.service_start, args.service_end, service_id="DAILY",
        )
        all_stops += s
        all_routes += r
        all_trips += t
        all_stop_times += st
        all_freq += fr
        all_shapes += sh
        print(f"[{mode}] routes={len(r)} stops={len(s)} stop_times={len(st)}")

    if not found_any:
        raise SystemExit(f"No bus_shapefiles/ or metro_shapefiles/ found for city={args.city!r} under {args.data_root}")

    os.makedirs(args.out, exist_ok=True)

    agency_name = args.agency_name or f"{args.city.title()} Public Transport (CPTOND-2025 reconstruction)"
    write_csv(os.path.join(args.out, "agency.txt"), [{
        "agency_id": "CPTOND",
        "agency_name": agency_name,
        "agency_url": "https://doi.org/10.6084/m9.figshare.29377427",
        "agency_timezone": "Asia/Shanghai",
        "agency_lang": "zh",
    }], ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang"])

    write_csv(os.path.join(args.out, "calendar.txt"), [{
        "service_id": "DAILY", "monday": 1, "tuesday": 1, "wednesday": 1,
        "thursday": 1, "friday": 1, "saturday": 1, "sunday": 1,
        "start_date": "20250101", "end_date": "20251231",
    }], ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
         "saturday", "sunday", "start_date", "end_date"])

    write_csv(os.path.join(args.out, "stops.txt"), all_stops,
              ["stop_id", "stop_name", "stop_lat", "stop_lon"])
    write_csv(os.path.join(args.out, "routes.txt"), all_routes,
              ["route_id", "route_short_name", "route_long_name", "route_type"])
    write_csv(os.path.join(args.out, "trips.txt"), all_trips,
              ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"])
    write_csv(os.path.join(args.out, "stop_times.txt"), all_stop_times,
              ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
    write_csv(os.path.join(args.out, "frequencies.txt"), all_freq,
              ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"])
    write_csv(os.path.join(args.out, "shapes.txt"), all_shapes,
              ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])

    print(f"Wrote GTFS feed to {args.out}")
    print(f"  routes={len(all_routes)} stops={len(all_stops)} trips={len(all_trips)} "
          f"stop_times={len(all_stop_times)} frequencies={len(all_freq)} shape_points={len(all_shapes)}")


if __name__ == "__main__":
    main()
