"""Ground the bus headways of the non-Shanghai cities of the Shanghai metro-area feed on real figures and on each city's real fleet.

The builder gives every ordinary route outside Shanghai the same flat 20/25/30 min profile (an assumption transferred from Suzhou). Here, per city:

 1. KNOWN lines get their published headways (FIXED_LINES: operator notices, municipal notices, Baidu Baike line pages, local portals; sources in the table).
 2. RURAL routes (median stop more than RURAL_KM from the city centre) get at least RURAL_FLOOR_MIN peak / RURAL_FLOOR_OFF off-peak: sampled evidence is village
    lines every 40-90 min (Suzhou), town lines every 30-60 min (Huzhou), and some lines only twice a day, so the flat 20-30 min was too frequent for them.
 3. All other routes keep their class relativities and are rescaled by one factor per city so the buses the feed needs at 08:00 on a weekday
    (trip time x 1.15 layover / headway) equal the real fleet in service (fleet x (1 - 0.08 maintenance - 0.09 spare)) minus the buses of the fixed routes.
    Within that set, routes near the city centre are more frequent than outer ones (median stop distance: <= 5 km weight 0.7, 5-12 km 1.0, > 12 km 1.4).

Cities without a found fleet (Nantong, Jiaxing) keep the generic scale (k = 1) but still get steps 1 and 2; Shanghai is untouched.

    python calibrate_other_city_buses.py <feed_dir>      # rewrites <feed_dir>/frequencies.txt from frequencies.orig.txt (kept as the backup)
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_SERVICE = 1 - 0.08 - 0.09
# prefix: (centre lat, lon, fleet or None, scope km for the fleet match, source)
CITIES = {
    "cz": (31.81, 119.97, 2004, 25, "Changzhou Public Transport Group profile, 2024: 2,004 vehicles, 291 lines (urban)"),
    "wx": (31.49, 120.31, 2599 * 343 / 270, 40, "Wuxi Transport Industry Group 2,599 buses on 270 lines (Mar 2024) scaled to the 343 routes of the 2024 bulletin"),
    "hz": (30.89, 120.09, 1083, 25, "Huzhou bus company 1,083 vehicles, 197 lines (Mar 2024, rating report; urban)"),
    "zs": (30.02, 122.11, 772, 60, "Zhoushan Baidu Baike (Dec 2023): 155 lines, 772 buses"),
    "sz": (31.30, 120.59, 5316, 40, "Suzhou 2024 statistical bulletin: 5,316 buses on 791 urban routes"),
    "nb": (29.87, 121.55, 9000, 40, "Ningbo network summary: 9,000+ buses, 1,200+ lines (year not stated)"),
    "nt": (32.01, 120.86, 1159 + 281, 40, "Nantong Public Transport Group 1,159 vehicles on 111 lines (company profile, Oct 2025) + its district subsidiary Feihe 281 vehicles on 45 lines (Baidu Baike, Dec 2024)"),
    "jx": (30.75, 120.76, 93 * 6.9, 12, "ESTIMATE: 93 urban lines (Baidu Baike, Aug 2023) x 6.9 vehicles per line (Changzhou group: 2,004 vehicles on 291 lines); the rural and intercity lines (90) belong to other operators and are not counted"),
}
# (city prefix, regex on the Chinese route name, peak min, off-peak min, source)
FIXED_LINES = [
    ("cz", r"^B1路\(", 5.5, 10, "Baidu Baike B1: peak 5-6, off-peak 8-12 (Oct 2024)"),
    ("cz", r"^B2路\(", 4.5, 7, "Baidu Baike B2: peak 4-5, off-peak 6-8"),
    ("cz", r"^B22路\(", 6, 8.5, "Bendibao: peak 6, off-peak 7-10"),
    ("cz", r"^B21路\(", 10, 20, "Baidu Baike B21: peak 10, off-peak 20"),
    ("nt", r"^56路\(", 32.5, 32.5, "Nantong notice: 30-35"),
    ("nt", r"^89路\(", 26.5, 26.5, "Nantong notice: 23-30"),
    ("nt", r"^61路\(", 35, 35, "Nantong notice: 30-40"),
    ("nt", r"^128路\(", 25, 25, "Baidu Baike (Mar 2024): 20-30"),
    ("nt", r"^99路\(", 25, 30, "Nantong notice: peak 25, off-peak 30"),
    ("nt", r"^45路\(", 15, 30, "Nantong notice: 15-30"),
    ("nt", r"^D28路\(", 15, 30, "Nantong notice (Nov 2024): 15-30"),
    ("sz", r"^116路\(", 15, 28, "Suzhou Industrial Park operator (2025): peak 12-18, off-peak 25-30"),
    ("nb", r"^825路\(", 20, 26, "Ningbo notice (2026): peak 15-25, off-peak 23-29"),
    ("hz", r"^55路\(", 6, 17, "Huzhou portal: 6-17"),
    ("hz", r"^801路\(", 20, 30, "Zhejiang news 2021: peak 20, off-peak 30"),
]
# Line-class detection by name/number (Chinese route name): relative headway weight (smaller = more frequent) and the shortest headway allowed, in seconds.
# BRT-like / fast / trunk lines are known to run more often than ordinary ones; the weights are assumptions anchored on the sampled lines above
# (Changzhou B1/B2 4.5-5.5 min peak, B22 6, B21 10 vs ordinary lines 20-35 min). Applied before the fleet rescale, so the fleet total is respected.
CLASSES = (
    (re.compile(r"^B\d+路|快速公交|BRT|大站快|快1|^快\d"), 0.45, 300, "brt_like"),
    (re.compile(r"快线"), 0.75, 360, "fast"),
    (re.compile(r"^\d{1,2}路\("), 0.85, 240, "trunk_1_2_digit"),
)
RURAL_KM = 25.0
RURAL_FLOOR_MIN, RURAL_FLOOR_OFF = 40.0, 60.0          # inter-town rural routes
VILLAGE_FLOOR_MIN, VILLAGE_FLOOR_OFF = 60.0, 90.0      # village / town-internal / community rural routes (numbers 200-399 and 900-999, or village names)
VILLAGE_RE = re.compile(r"(?<!\d)(2\d\d|3\d\d|9\d\d)路|社区|村|镇|微\d")
TIERS = ((3.0, 0.55), (5.0, 0.7), (12.0, 1.0), (1e9, 1.4))   # very central routes (median stop within 3 km) are the most frequent
HW_MIN, HW_MAX = 240, 5400


def sec(x):
    h, m, s = (int(v) for v in str(x).split(":"))
    return h * 3600 + m * 60 + s


def is_peak(mid_s):
    h = mid_s / 3600
    return 7 <= h < 9 or 16.5 <= h < 19


def main(feed_dir):
    d = Path(feed_dir)
    orig = d / "frequencies.orig.txt"
    if not orig.exists():
        shutil.copy(d / "frequencies.txt", orig)
    f = pd.read_csv(orig)
    routes = pd.read_csv(d / "routes.txt", usecols=["route_id", "route_long_name", "route_type"])
    trips = pd.read_csv(d / "trips.txt", usecols=["route_id", "trip_id", "service_id"])
    st = pd.read_csv(d / "stop_times.txt", usecols=["trip_id", "stop_id", "arrival_time", "departure_time"])
    stops = pd.read_csv(d / "stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"]).set_index("stop_id")
    g = st.groupby("trip_id")
    dur = (g.arrival_time.last().map(sec) - g.departure_time.first().map(sec)).rename("dur")
    trips = trips.merge(routes, on="route_id").merge(dur, left_on="trip_id", right_index=True)
    trips["pfx"] = trips.route_id.str.split("_").str[0]
    f["s"], f["e"] = f.start_time.map(sec), f.end_time.map(sec)
    f["peak"] = ((f.s + f.e) / 2).map(is_peak)
    f["late"] = f.e > 21 * 3600
    f = f.merge(trips[["trip_id", "route_id", "pfx", "route_type", "service_id", "dur", "route_long_name"]].drop_duplicates("trip_id"), on="trip_id", how="left")
    f["hw_new"] = f.headway_secs.astype(float)
    report = {}
    for pfx, (la0, lo0, fleet, scope, src) in CITIES.items():
        sel = (f.pfx == pfx) & (f.route_type == 3)
        t = f[sel]
        if t.empty:
            continue
        rids = t.route_id.unique()
        rs = st.merge(trips[trips.route_id.isin(rids)][["trip_id", "route_id"]], on="trip_id")[["route_id", "stop_id"]].drop_duplicates().join(stops, on="stop_id")
        rs["km"] = np.hypot((rs.stop_lat - la0) * 111.0, (rs.stop_lon - lo0) * 111.0 * np.cos(np.radians(la0)))
        med = rs.groupby("route_id").km.median()
        cn = routes.set_index("route_id").route_long_name.astype(str).str.split("/").str[-1].str.strip()
        fixed = {}          # route_id -> (peak min, off min)
        for p_, rx, pk, off, _ in FIXED_LINES:
            if p_ != pfx:
                continue
            for r in rids:
                if re.search(rx, cn.get(r, "")):
                    fixed[r] = (pk, off)
        n_known = len(fixed)
        rural = [r for r in rids if r not in fixed and med.get(r, 0) > RURAL_KM]
        # fixed headways per row
        def fixed_hw(row):
            if row.route_id in fixed:
                pk, off = fixed[row.route_id]
                return (pk if row.peak else off * (1.2 if row.late else 1.0)) * 60
            return None
        # rural floor
        is_rural = f.route_id.isin(rural)
        village = f.route_id.map(lambda r: bool(VILLAGE_RE.search(cn.get(r, "")))).to_numpy()
        floor = np.where(village, np.where(f.peak, VILLAGE_FLOOR_MIN * 60, VILLAGE_FLOOR_OFF * 60), np.where(f.peak, RURAL_FLOOR_MIN * 60, RURAL_FLOOR_OFF * 60))
        f.loc[sel & is_rural, "hw_new"] = np.maximum(f.loc[sel & is_rural, "headway_secs"], floor[(sel & is_rural).to_numpy()])
        known_rows = sel & f.route_id.isin(fixed)
        if known_rows.any():
            f.loc[known_rows, "hw_new"] = f[known_rows].apply(fixed_hw, axis=1)
        free = sel & ~f.route_id.isin(fixed) & ~is_rural
        # buses at 08:00 weekday
        at8 = (f.s <= 8 * 3600 + 1800) & (f.e > 8 * 3600 + 1800) & (f.service_id == "WD")
        w = f.route_id.map(med).map(lambda k: next((wt for lim, wt in TIERS if k <= lim), 1.0) if pd.notna(k) else 1.0)
        cls = {r: next(((wt, mn, nm) for rx, wt, mn, nm in CLASSES if rx.search(cn.get(r, ""))), (1.0, HW_MIN, "ordinary")) for r in rids}
        w = w * f.route_id.map(lambda r: cls.get(r, (1.0,))[0])
        minhw = f.route_id.map(lambda r: cls.get(r, (1.0, HW_MIN))[1]).fillna(HW_MIN)
        n_class = pd.Series([c[2] for r, c in cls.items() if r not in fixed and r not in rural]).value_counts().to_dict()
        k = 1.0
        buses_fixed = (f.dur * 1.15 / f.hw_new)[sel & at8 & ~free & (f.route_id.map(med) <= (scope or 1e9))].sum()
        base = (f.dur * 1.15 / (f.headway_secs * w))[sel & at8 & free & (f.route_id.map(med) <= (scope or 1e9))].sum()
        target = fleet * IN_SERVICE if fleet else None
        if target:
            need = max(target - buses_fixed, 0.15 * target)
            k = base / need
            k_min = 0.5   # never compress the generic headways by more than half, even if the fleet would allow it (the fleet figure may include buses outside the urban scope)
            for _ in range(8):
                k = max(k, k_min)
                hw = np.minimum(np.maximum(f.headway_secs * w * k, minhw), HW_MAX)
                b = (f.dur * 1.15 / hw)[sel & at8 & free & (f.route_id.map(med) <= scope)].sum()
                k *= b / need
        f.loc[free, "hw_new"] = np.minimum(np.maximum(f.loc[free, "headway_secs"] * w[free] * k, minhw[free]), HW_MAX)
        after = (f.dur * 1.15 / f.hw_new)[sel & at8 & (f.route_id.map(med) <= (scope or 1e9))].sum()
        pk = sel & f.peak
        report[pfx] = {"k": round(k, 3), "fleet_in_service_target": round(target) if target else None, "feed_buses_in_scope_after": round(after),
                       "known_lines": n_known, "k_capped_at_0.5": bool(k <= 0.5001), "free_routes_by_class": n_class, "rural_routes_floored": len(rural), "routes": int(len(rids)),
                       "mean_peak_headway_min_before": round(f[pk].headway_secs.mean() / 60, 1), "mean_peak_headway_min_after": round(f[pk].hw_new.mean() / 60, 1),
                       "mean_offpeak_headway_min_after": round(f[sel & ~f.peak].hw_new.mean() / 60, 1), "source": src}
    out = f[["trip_id", "start_time", "end_time"]].copy()
    out["headway_secs"] = f.hw_new.round().astype(int)
    out["exact_times"] = f.exact_times
    out.to_csv(d / "frequencies.txt", index=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assumed_feeds"))
    from stagger_phase import stagger_feed   # independent lines get their own phase (see stagger_phase.py); metro untouched
    print("staggered window starts:", stagger_feed(d))
    (d / "bus_calibration.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
