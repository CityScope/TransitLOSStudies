"""Build a frequency-based GTFS feed for the Shanghai metropolitan area.

Covers the "1+8" Shanghai Metropolitan Area (上海大都市圈): Shanghai plus
Suzhou, Wuxi, Changzhou, Nantong, Jiaxing, Huzhou, Ningbo and Zhoushan --
every bus and metro route CPTOND-2025 has for those nine cities, not just the
study AOI -- plus the Shanghai Maglev (not in CPTOND, built from OSM).

CPTOND-2025 gives route existence, ordered stops and route geometry, but no
timetable. Headways, service spans, time bands and running speeds below are
researched values; every number cites its source in
`shanghai/gtfs/sources/` (see `GTFS_CONSTRUCTION.md` there for the full
write-up and what is data vs. assumption).

Usage (from `city_science_network/`):
    uv run python china/CPTOND-2025/code/build_shanghai_metro_area_gtfs.py \\
        --out shanghai/gtfs/shanghai_metro_area
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from shapely.prepared import prep

from build_gtfs_from_cptond import along_line_km, haversine_km, write_csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "dataset")
SOURCES = os.path.join(HERE, "..", "..", "..", "shanghai", "gtfs", "sources")

# (CPTOND folder, file prefix, id prefix, agency name)
CITIES = [
    ("Shanghai", "shanghai", "sh", "Shanghai"),
    ("Suzhou", "suzhou", "sz", "Suzhou"),
    ("Wuxi", "wuxi", "wx", "Wuxi"),
    ("Changzhou", "changzhou", "cz", "Changzhou"),
    ("Nantong", "nantong", "nt", "Nantong"),
    ("Jiaxing", "jiaxing", "jx", "Jiaxing"),
    ("Huzhou", "huzhou", "hz", "Huzhou"),
    ("ningbo", "ningbo", "nb", "Ningbo"),
    ("Zhoushan", "zhoushan", "zs", "Zhoushan"),
]

# ---------------------------------------------------------------------------
# Time bands (minutes after midnight; > 1440 = after midnight of the next day).
# Peaks 07:00-09:00 / 17:00-19:00: the peak columns of the Shanghai Metro
# headway table and the bus/metro sources below all use peaks of this shape
# (Changzhou/Nantong/Wuxi metro 07:00/07:30-09:00 and 16:30-18:30).
WD_BANDS = [(0, 420, "early"), (420, 540, "am"), (540, 1020, "midday"),
            (1020, 1140, "pm"), (1140, 1260, "evening"), (1260, 1800, "late")]
WE_BANDS = [(0, 420, "early"), (420, 1140, "weekend_day"),
            (1140, 1260, "evening"), (1260, 1800, "late")]
SERVICES = {  # service_id -> (mon..sun)
    "WD": (1, 1, 1, 1, 1, 0, 0),
    "WE": (0, 0, 0, 0, 0, 1, 1),
    "FRISAT": (0, 0, 0, 0, 1, 1, 0),
    "SUNTHU": (1, 1, 1, 1, 0, 0, 1),
}
CALENDAR_START, CALENDAR_END = "20250101", "20271231"


def band_profile(am, midday, pm, evening, late, early=None, weekend_day=None):
    return {"early": early if early is not None else evening, "am": am, "midday": midday,
            "pm": pm, "evening": evening, "late": late,
            "weekend_day": weekend_day if weekend_day is not None else midday}


def banded(profile, open_min, close_min, bands):
    out = []
    for s, e, band in bands:
        s2, e2 = max(s, open_min), min(e, close_min)
        hw = profile.get(band)
        if s2 < e2 and hw:
            out.append((s2, e2, hw))
    return out


# ---------------------------------------------------------------------------
# Buses
# ---------------------------------------------------------------------------
# Shanghai zone classes follow the 2024 bus service standard (上海市公共汽车和
# 电车客运服务规范, art. 22-23, sources/sh_bus_service_spec_2024.html): a
# route is "central" when both terminals are inside the S20 Outer Ring, or it
# crosses the ring with >= 2/3 of its length inside. Headway ceilings there:
# central <= 8 min peak / <= 20 other; trunk (骨干线) <= 5 / <= 15; all other
# routes <= 25. Values below were calibrated so the buses needed to run them
# (route length / zone speed * 1.15 layover / headway) match ~90% of the real
# fleet (~17,000 buses, 2024 annual report): 8/12/15 min peak -> ~15,200.
SH_BUS = {
    "central": band_profile(am=8, midday=12, pm=8, evening=12, late=20),
    "radial": band_profile(am=12, midday=15, pm=12, evening=15, late=25),
    "suburban": band_profile(am=15, midday=20, pm=15, evening=20, late=25),
    "trunk": band_profile(am=5, midday=10, pm=5, evening=10, late=15),
}
# Other metro-area cities: Suzhou's 2024 statistical bulletin gives 5,316
# buses on 791 urban routes; with CPTOND's Suzhou mean route length (17.5 km)
# at 20 km/h and the same layover/availability, that is ~20 min at peak.
OTHER_BUS = band_profile(am=20, midday=25, pm=20, evening=25, late=30)
FERRY = band_profile(am=15, midday=20, pm=15, evening=20, late=25)  # Wikipedia "Shanghai Ferry": 10-30 min
FLAT = lambda hw: {b: hw for b in ("early", "am", "midday", "pm", "evening", "late", "weekend_day")}

SH_SPEED_INSIDE_RING_KMH = 16.0   # 2021 central-city peak average 16.1/15.8 km/h
SH_SPEED_OUTSIDE_RING_KMH = 21.0  # gives the citywide 19.2 km/h average (official 19.6/19.2)
OTHER_BUS_SPEED_KMH = 20.0
LAYOVER = 1.15

NIGHT_RE = re.compile(r"夜宵|夜班|夜间|通宵")
OVERNIGHT_RE = re.compile(r"夜宵|通宵")
PEAK_ONLY_RE = re.compile(r"高峰")
TRUNK_RE = re.compile(r"中运量|^71路")
CUSTOM_RE = re.compile(r"^DZ|定制", re.IGNORECASE)
FIXED_RE = re.compile(r"定班|村村通|最后一公里")
VILLAGE_RE = re.compile(r"乡村|便民|社区|微巴|微公交|^微\d|^[^\d(]*微\d|^W\d")
RAIL_IN_BUS_RE = re.compile(r"城际铁路")
NUMBERED_RE = re.compile(r"\d+路")


def parse_hhmm(v):
    v = (str(v) if v is not None else "").strip()
    if not re.fullmatch(r"\d{3,4}", v):
        return None
    v = v.zfill(4)
    h, m = int(v[:2]), int(v[2:])
    return h * 60 + m if h < 24 and m < 60 else None


def cptond_span(row):
    s, e = parse_hhmm(row.get("start_time")), parse_hhmm(row.get("end_time"))
    if s is None or e is None or s == e:
        return None
    if e < s:
        e += 1440
    return s, e


# Shanghai numbered routes: zh.wikipedia "上海市公交线路列表 (数字线路)"
# [sources/zhwiki_shanghai_bus_routes_numbered.html, parsed by
# parse_zhwiki_shanghai_routes.py]. Numbering convention stated there:
# 2xx = peak-only lines (高峰线) unless listed under 全日运营线路; 3xx = night
# lines (夜宵线), 23:00-06:00. It also gives per-direction first/last times and,
# for night lines, the headway ("30分钟/班").
PEAK_ONLY_WINDOWS = [(330, 570), (930, 1170)]  # 05:30-09:30, 15:30-19:30
SH_PLAIN_NUMBER_RE = re.compile(r"^(\d{1,4})路")
ZHWIKI = defaultdict(list)


def load_zhwiki():
    path = os.path.join(SOURCES, "zhwiki_shanghai_numbered_routes_parsed.csv")
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ZHWIKI[row["number"]].append(row)


def zhwiki_row(name):
    m = SH_PLAIN_NUMBER_RE.match(name)
    rows = ZHWIKI.get(m.group(1)) if m else None
    if not rows:
        return None
    interval = "区间" in name
    return next((r for r in rows if ("区间" in r["label"]) == interval), rows[0])


def parse_ranges(text, night=False):
    out = []
    for part in filter(None, (text or "").split(";")):
        s, e = (int(h) * 60 + int(m) for h, m in (x.split(":") for x in part.split("-")))
        if e <= s:
            e += 1440
        if night and s < 720:  # 00:25-04:20 belongs to the previous evening's service day
            s, e = s + 1440, e + 1440
        out.append((s, e))
    return out


def zhwiki_ranges(row, route_cn, night=False):
    """Departure-time ranges from this route's origin terminal (the
    direction is picked by matching CPTOND's origin name to the listed
    left/right terminal)."""
    parts = row["raw"].split(" | ")
    if "⇆" not in parts:
        return parse_ranges(row["times_a"], night)
    i = parts.index("⇆")
    left, right = parts[i - 1], parts[i + 1]
    m = re.match(r"^[^(]*\((.*?)--", route_cn)
    origin = m.group(1) if m else ""
    score = lambda term: len(set(origin) & set(re.sub(r"[\d:–\-\s]", "", term)))
    return parse_ranges(row["times_a"] if score(left) >= score(right) else row["times_b"], night)


def bus_service(city_key, route_cn, type_en, zone, span, log=None):
    """-> (category, {service_id: [(start_min, end_min, headway_min)]})"""
    name = route_cn or ""
    t = (type_en or "").strip()
    log = log if log is not None else defaultdict(int)
    wiki = zhwiki_row(name) if city_key == "sh" else None
    m = SH_PLAIN_NUMBER_RE.match(name) if city_key == "sh" else None
    num = int(m.group(1)) if m else None
    wiki_tags = (wiki["label"] + " " + wiki["section"]) if wiki else ""

    wiki_first = zhwiki_ranges(wiki, name) if wiki and wiki["times_a"] else []
    first_dep = wiki_first[0][0] if wiki_first else (span[0] if span else None)
    starts_at_night = first_dep is not None and first_dep >= 1260  # first departure 21:00 or later
    if (NIGHT_RE.search(name) or "夜宵" in wiki_tags or (num is not None and 300 <= num <= 399)
            or starts_at_night):
        ranges = zhwiki_ranges(wiki, name, night=True) if wiki and wiki["times_a"] else []
        if ranges:
            log["night span: zh.wikipedia"] += 1
        elif span:
            s, e = span
            ranges = [(s + 1440, e + 1440) if s < 720 else (s, e)]
            log["night span: CPTOND"] += 1
        elif city_key == "sh" or OVERNIGHT_RE.search(name):
            ranges = [(1380, 1740)]  # standard art. 22: night lines 23:00-05:00
            log["night span: standard default (23:00-05:00)"] += 1
        else:
            # Outside Shanghai 夜班/夜间 routes are evening services (CPTOND
            # times: mostly ~18:00-21:00): use this city's median of those.
            ranges = [_EVENING_DEFAULT_SPAN.get(city_key, (1110, 1260))]
            log["night span: city median of evening (夜班) routes"] += 1
        if wiki and wiki["headway_min"]:
            hw = float(wiki["headway_min"])
            log["night headway: zh.wikipedia"] += 1
        else:
            hw = 40 if NUMBERED_RE.search(name) else 60  # art. 23 ceilings
            log["night headway: standard ceiling"] += 1
        w = [(s, e, hw) for s, e in ranges]
        overnight = any(s >= 1260 or e > 1440 for s, e in ranges)
        return ("night" if overnight else "evening"), {"WD": w, "WE": w}

    all_day_2xx = "全日运营" in wiki_tags
    if PEAK_ONLY_RE.search(name) or "高峰线" in wiki_tags or (num is not None and 200 <= num <= 299 and not all_day_2xx):
        ranges = zhwiki_ranges(wiki, name) if wiki and wiki["times_a"] else []
        # keep only the peak windows inside the listed hours (a few 高峰线
        # are listed with one all-day range, e.g. 777路 06:45-19:00)
        w = [(max(s, ps), min(e, pe), 8) for s, e in ranges for ps, pe in PEAK_ONLY_WINDOWS
             if max(s, ps) < min(e, pe)]
        if w:
            log["peak-only span: zh.wikipedia"] += 1
        else:
            am, pm = (360, 510, 8), (990, 1140, 8)  # art. 22: 06:00-08:30 / 16:30-19:00, <= 8 min
            w = [am] if re.search("早高峰|上午高峰", name) else [pm] if re.search("晚高峰|下午高峰", name) else [am, pm]
            log["peak-only span: standard default"] += 1
        return "peak_only", {"WD": w}

    if wiki and wiki["times_a"]:
        ranges = zhwiki_ranges(wiki, name)
        if ranges:
            span = (ranges[0][0], ranges[-1][1])
            log["span: zh.wikipedia"] += 1
    elif span:
        log["span: CPTOND"] += 1
    if CUSTOM_RE.search(name) and not (city_key == "sh" and TRUNK_RE.search(name)):
        w = [(420, 540, 20), (1020, 1140, 20)]  # customized commuter routes, weekday peaks only
        return "customized", {"WD": w}
    if t == "Ferry":
        profile, cat = FERRY, "ferry"
    elif t == "Tourist line":
        profile, cat = FLAT(20), "tourist"
        span = span or (480, 1080)
    elif t == "Airport Limousine Bus":
        profile, cat = FLAT(20), "airport"
    elif FIXED_RE.search(name):
        profile, cat = FLAT(60), "fixed_schedule"  # art. 23: timetabled, <= 60 min
    elif t == "Community car" or VILLAGE_RE.search(name):
        profile, cat = FLAT(30), "community_village"
    elif city_key == "sh" and TRUNK_RE.search(name):
        profile, cat = SH_BUS["trunk"], "trunk"
    elif city_key == "sh":
        profile, cat = SH_BUS[zone], zone
    else:
        profile, cat = OTHER_BUS, "regular"
    if span is None:
        span = default_bus_span(city_key, zone, name)
        log["span: default"] += 1
    s, e = span
    return cat, {"WD": banded(profile, s, e, WD_BANDS), "WE": banded(profile, s, e, WE_BANDS)}


_OTHER_CITY_DEFAULT_SPAN = {}
_EVENING_DEFAULT_SPAN = {}


def default_bus_span(city_key, zone, name):
    if city_key == "sh":  # art. 22 first/last bus requirements
        if zone == "central":
            return (330, 1350) if NUMBERED_RE.search(name) else (360, 1260)
        return (360, 1260) if zone == "radial" else (360, 1140)
    return _OTHER_CITY_DEFAULT_SPAN[city_key]


# ---------------------------------------------------------------------------
# Metro
# ---------------------------------------------------------------------------
def mm_ss(v):
    m, s = v.split("'") if "'" in v else (v, "0")
    return int(m) + int(s or 0) / 60


def sh_line(am, mid, pm, we, lo, hi):
    """Shanghai Metro table row -> band profile. 'Other hours' is a range:
    lower bound for 19:00-21:00, upper bound after 21:00, midpoint before 07:00."""
    am, mid, pm, we, lo, hi = (mm_ss(x) for x in (am, mid, pm, we, lo, hi))
    return band_profile(am=am, midday=mid, pm=pm, evening=lo, late=hi,
                        early=(lo + hi) / 2, weekend_day=we)


# Wikipedia "Shanghai Metro", "Headways by line", inner sections, updated July
# 2023 (sources/wikipedia_shanghai_metro.wikitext):
# morning peak, mid-day, evening peak, weekend daytime, other hours low-high.
SH_METRO = {
    1: sh_line("2'30", "4'00", "3'00", "4'00", "4'00", "12'00"),
    2: sh_line("2'30", "4'00", "3'00", "3'40", "5'00", "11'00"),
    3: sh_line("5'00", "7'00", "5'00", "4'00", "6'30", "10'00"),
    4: sh_line("3'10", "8'00", "5'00", "6'30", "5'00", "12'00"),
    5: sh_line("2'30", "4'30", "3'00", "4'00", "7'30", "10'00"),
    6: sh_line("2'00", "4'30", "2'30", "3'20", "5'00", "10'00"),
    7: sh_line("1'55", "5'30", "2'44", "5'00", "5'00", "8'00"),
    8: sh_line("2'00", "4'20", "2'45", "3'40", "5'00", "6'00"),
    9: sh_line("1'50", "5'30", "2'30", "5'00", "6'00", "10'00"),
    10: sh_line("2'30", "4'00", "3'00", "3'45", "5'00", "8'00"),
    11: sh_line("2'00", "6'00", "3'00", "5'00", "6'00", "8'00"),
    12: sh_line("2'30", "6'00", "3'45", "6'00", "6'00", "10'00"),
    13: sh_line("2'30", "6'00", "3'00", "5'00", "6'00", "10'00"),
    14: sh_line("3'20", "6'00", "4'00", "5'00", "5'00", "12'00"),
    15: sh_line("3'40", "8'00", "4'30", "6'00", "5'00", "13'00"),
    16: sh_line("3'30", "7'00", "4'00", "5'00", "5'00", "10'00"),
    17: sh_line("3'30", "10'00", "4'00", "7'00", "8'00", "10'00"),
    18: sh_line("3'00", "7'00", "3'30", "6'30", "6'30", "12'00"),
    "pujiang": sh_line("3'20", "6'30", "3'50", "6'00", "9'00", "9'00"),
}
# Same table, outer sections (maximum interval) -- used for each branch of the
# branched lines; the shared trunk gets an extra short-turn service so that
# its combined frequency equals the inner-section value.
SH_METRO_OUTER = {
    5: sh_line("7'30", "9'00", "9'00", "8'00", "7'30", "10'00"),
    10: sh_line("9'00", "12'00", "9'00", "10'00", "6'00", "14'00"),
    11: sh_line("10'00", "12'00", "7'00", "10'00", "11'00", "18'00"),
}
SH_METRO_HOURS = (330, 1380)  # Wikipedia: most stations open 5:00-6:00, close 22:30-23:00
SH_METRO_FRISAT_EXTENDED = {1, 2, 7, 9, 10, 13}  # +1 h on Fri/Sat (Wikipedia, extended hours)


def other_line(peak, off, late=10.0, am=None, pm=None, we=None):
    return band_profile(am=am or peak, midday=off, pm=pm or peak, evening=off, late=late,
                        early=off, weekend_day=we or off)


M = lambda m, s=0: m + s / 60  # minutes + seconds
# Other cities' metro headways, each from a page archived in
# sources/metro_area/ (file named in the comment). Lines with no published
# figure use the city default, flagged as such.
OTHER_METRO = {
    "sz": {
        1: other_line(4, 8, am=3, we=M(6, 50)),               # suzhou_metro_schedule_adjust_163 (Apr 2022, COVID-era; zhwiki_suzhou_line1 2018 table: <= 4 off-peak)
        2: other_line(M(5, 20), M(6, 45), am=M(4, 30), we=M(5, 20)),  # zhwiki_suzhou_line2
        4: other_line(6, 8, am=4.5, we=8),                     # suzhou_metro_schedule_adjust_163 (2022)
        7: other_line(6, 7, we=7),                             # suzhou_metro_line7_thepaper
        11: other_line(6.5, 8, we=8),                          # zhwiki_suzhou_line11
        "default": other_line(6, 8),  # Lines 3/5/6/8: 2022 notice gives 8 all day (Lines 3/5); peak 6 assumed
        "tram": other_line(10, 12, late=15),                   # no source found: assumption
    },
    "wx": {  # wuxi_metro_line2_interval_bendibao (2020 notice covering Lines 1 and 2)
        1: other_line(6, 8, late=9), 2: other_line(6, 8, late=9),
        "default": other_line(6, 8, late=9),  # Lines 3/4/S1: no archived figure, Lines 1/2 values used
    },
    "nb": {  # zhwiki_ningbo_line1..5 (operator timetables 2021-2023)
        1: band_profile(am=M(3, 50), midday=M(7, 40), pm=M(3, 50), evening=M(7, 40), late=M(7, 40),
                        early=M(7, 40), weekend_day=M(5, 30)),
        2: other_line(M(4, 20), M(4, 54), late=10, we=M(6, 20)),
        3: other_line(9, 9, late=10, we=M(6, 30)),
        4: other_line(M(4, 25), M(6, 20), late=10, we=M(6, 25)),
        5: other_line(M(4, 58), M(5, 52), late=M(7, 40), we=M(5, 52)),
        "default": other_line(M(4, 25), M(6, 20)),
    },
    "cz": {"default": other_line(7, 8, late=10, we=7)},  # changzhou_metro_line1_timetable_bendibao (Line 2 assumed same)
    "nt": {"default": other_line(7, 8, late=M(14, 50), we=8)},  # nantong_metro_line1_bendibao + zhwiki_nantong_line1
}
# Short-turn (小交路) services on top of the full-line trains (zhwiki_ningbo_line1/3):
# (city, line) -> (first station, last station, band profile of the short-turn trains alone).
# Line 1 off-peak: 3'50 on the core = full-line 7'40 + short-turn 7'40.
# Line 3 until 20:00: 4'30 on the core = full-line 9' + short-turn 9'.
SHORT_TURNS = {
    ("nb", 1): ("望春桥", "东环南路", {"midday": M(7, 40), "evening": M(7, 40)}),
    ("nb", 3): ("大通桥", "明辉路", {"am": 9, "midday": 9, "pm": 9, "evening": 9}),
}
OTHER_METRO_HOURS = {"cz": (360, 1320), "nt": (360, 1320), "nb": (345, 1320), "sz": (340, 1350),
                     "default": (360, 1350)}
METRO_SPEED_KMH = 35.0
TRAM_SPEED_KMH = 18.0

# Maglev (Wikipedia "Shanghai maglev train" + operator timetable pages):
# 30 km in 8 min 10 s (300 km/h cruise since May 2021), 06:45-21:42,
# every 20 min except 15 min between the first two trains.
MAGLEV_RUN_SECONDS = 8 * 60 + 10
MAGLEV_WINDOWS = [(405, 420, 15), (420, 1302, 20)]


def metro_line_key(name):
    if "浦江线" in name:
        return "pujiang"
    if "有轨电车" in name or "智轨" in name:
        return "tram"
    m = re.search(r"(\d+)号线", name)
    return int(m.group(1)) if m else None


def metro_drop_reason(city_key, name, status):
    if str(status) != "1":
        return f"status {status} (not operating in CPTOND)"
    if "夜班" in name:
        return "night-shift trains (modelled separately)"
    if "大站车" in name:
        return "express variant (covered by the line headway)"
    if city_key == "sz" and name.startswith("地铁11号线"):
        return "duplicate of Shanghai Metro Line 11"
    return None


# ---------------------------------------------------------------------------
def fmt(minutes_or_seconds, seconds=False):
    s = int(round(minutes_or_seconds if seconds else minutes_or_seconds * 60))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load(city_dir, prefix, mode):
    d = os.path.join(DATASET, mode, "shapefiles", city_dir)
    if not os.path.isdir(d):
        return None
    stops = gpd.read_file(os.path.join(d, f"{prefix}_{mode}_stops.shp")).to_crs(4326)
    unique = gpd.read_file(os.path.join(d, f"{prefix}_{mode}_stops_unique.shp")).to_crs(4326)
    routes = gpd.read_file(os.path.join(d, f"{prefix}_{mode}_routes.shp")).to_crs(4326)
    return stops, unique, routes


class Feed:
    def __init__(self, simplify_deg):
        self.simplify = simplify_deg
        self.stops, self.routes, self.trips, self.stop_times = {}, [], [], []
        self.freqs, self.shapes, self.agencies = [], [], []
        self.log = defaultdict(lambda: defaultdict(int))
        self.seen_patterns = set()

    def add_shape(self, shape_id, line):
        geom = line.simplify(self.simplify, preserve_topology=False) if self.simplify else line
        for i, (lon, lat) in enumerate(geom.coords):
            self.shapes.append({"shape_id": shape_id, "shape_pt_lat": round(lat, 6),
                                "shape_pt_lon": round(lon, 6), "shape_pt_sequence": i})

    def add_trip(self, route_id, trip_id, service, shape_id, stop_ids, offsets_s, windows=None,
                 start_s=0, headsign=""):
        self.trips.append({"route_id": route_id, "service_id": service, "trip_id": trip_id,
                           "trip_headsign": headsign, "direction_id": 0, "shape_id": shape_id})
        for seq, (sid, off) in enumerate(zip(stop_ids, offsets_s)):
            t = fmt(start_s + off, seconds=True)
            self.stop_times.append({"trip_id": trip_id, "arrival_time": t, "departure_time": t,
                                    "stop_id": sid, "stop_sequence": seq})
        for s, e, hw in windows or []:
            self.freqs.append({"trip_id": trip_id, "start_time": fmt(s), "end_time": fmt(e),
                               "headway_secs": int(round(hw * 60)), "exact_times": 0})


def shape_is_ambiguous(line, lats, lons, dists_km, tol_km=0.1):
    """True if snapping any stop to its *nearest* point on `line`
    (pyGTFSHandler's rule: each stop independently, no ordering) could land it
    somewhere other than its in-order position `dists_km` -- i.e. the line
    passes that stop more than once (loops, out-and-back, circle lines).
    Such a shape would make pyGTFSHandler measure wrong inter-stop distances,
    so those trips are written without a shape (stop-to-stop geometry)."""
    c = np.asarray(line.coords)
    x1, y1, x2, y2 = c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1]
    dx, dy = x2 - x1, y2 - y1
    seg2 = np.where(dx * dx + dy * dy == 0, 1e-30, dx * dx + dy * dy)
    kx = np.cos(np.radians(c[:, 1].mean()))
    seg_km = np.hypot(dx * kx, dy) * 111.195
    cum = np.concatenate([[0], np.cumsum(seg_km)])
    scale = cum[-1] / max(dists_km[-1], 1e-9) if dists_km[-1] > 0 else 1.0
    for lat, lon, d in zip(lats, lons, dists_km):
        t = np.clip(((lon - x1) * dx + (lat - y1) * dy) / seg2, 0, 1)
        px, py = x1 + t * dx, y1 + t * dy
        dist = np.hypot((px - lon) * kx, py - lat)
        near = dist <= dist.min() + 1e-4  # any pass within ~10 m is a place the stop could snap to
        pos = cum[:-1][near] + t[near] * seg_km[near]
        if np.any(np.abs(pos - d * scale) > tol_km):
            return True
    return False


def along_route_km(line, lats, lons):
    """(cumulative stop distances km, shape usable?) -- along the real line,
    or stop-to-stop haversine when the line is ambiguous (see above)."""
    dists = along_line_km(line, lats, lons)
    if not shape_is_ambiguous(line, lats, lons, dists):
        return dists, True
    dists = [0.0]
    for i in range(1, len(lats)):
        dists.append(dists[-1] + haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i]))
    return dists, False


def route_names(r):
    en = (r.get("route_en") or "").strip()
    cn = (r.get("route_cn") or "").strip()
    short = (en or cn).split("(")[0].strip()[:40]
    return short, f"{en} / {cn}" if en else cn


def route_stop_sequences(stops_gdf):
    out = {}
    for rc, g in stops_gdf.sort_values(["route_cn", "sequence"]).groupby("route_cn", sort=False):
        out[rc] = g
    return out


def build_city(feed, city_dir, prefix, ck, agency, ring):
    feed.agencies.append({"agency_id": ck, "agency_name": f"{agency} public transport (CPTOND-2025 reconstruction)",
                          "agency_url": "https://doi.org/10.6084/m9.figshare.29377427",
                          "agency_timezone": "Asia/Shanghai", "agency_lang": "zh"})
    for mode in ("bus", "metro"):
        data = load(city_dir, prefix, mode)
        if data is None:
            continue
        stops, unique, routes = data
        for _, u in unique.iterrows():
            feed.stops[f"{ck}_{u['stop_id']}"] = {
                "stop_id": f"{ck}_{u['stop_id']}", "stop_name": u.get("stop_en") or u.get("stop_cn") or u["stop_id"],
                "stop_lat": round(u.geometry.y, 7), "stop_lon": round(u.geometry.x, 7)}
        seqs = route_stop_sequences(stops)
        if mode == "bus":
            if ck not in _OTHER_CITY_DEFAULT_SPAN and ck != "sh":
                spans = [sp for sp in (cptond_span(r) for _, r in routes.iterrows()) if sp]
                ss = sorted(s for s, _ in spans)
                es = sorted(e for _, e in spans)
                _OTHER_CITY_DEFAULT_SPAN[ck] = (ss[len(ss) // 2], es[len(es) // 2]) if spans else (360, 1260)
                ev = [sp for sp in (cptond_span(r) for _, r in routes.iterrows()
                                    if NIGHT_RE.search(r["route_cn"] or "") and not OVERNIGHT_RE.search(r["route_cn"] or ""))
                      if sp and sp[0] >= 900]
                if ev:
                    _EVENING_DEFAULT_SPAN[ck] = (sorted(a for a, _ in ev)[len(ev) // 2], sorted(b for _, b in ev)[len(ev) // 2])
            build_bus(feed, ck, routes, seqs, ring)
        else:
            build_metro(feed, ck, routes, seqs)


def build_bus(feed, ck, routes, seqs, ring):
    for _, r in routes.iterrows():
        rc = r["route_cn"]
        if str(r["status"]) != "1":
            feed.log[ck]["bus dropped: status 0 (suspended)"] += 1
            continue
        if RAIL_IN_BUS_RE.search(rc):
            feed.log[ck]["bus dropped: intercity rail line filed as bus in CPTOND"] += 1
            continue
        g = seqs.get(rc)
        if g is None or len(g) < 2 or r.geometry is None or r.geometry.geom_type != "LineString":
            feed.log[ck]["bus dropped: no stops/geometry"] += 1
            continue
        lats, lons = g.geometry.y.tolist(), g.geometry.x.tolist()
        pattern = tuple((round(a, 4), round(b, 4)) for a, b in zip(lats, lons))
        if pattern in feed.seen_patterns:
            feed.log[ck]["bus dropped: identical to a route already in the feed"] += 1
            continue
        feed.seen_patterns.add(pattern)

        zone = "regular"
        if ck == "sh":
            first_in = ring.contains(Point(lons[0], lats[0]))
            last_in = ring.contains(Point(lons[-1], lats[-1]))
            share_in = r.geometry.intersection(ring_poly).length / r.geometry.length
            zone = "central" if (first_in and last_in) or share_in >= 2 / 3 else (
                "radial" if first_in or last_in else "suburban")
        cat, services = bus_service(ck, rc, r.get("type_en"), zone, cptond_span(r), feed.log[ck])
        feed.log[ck][f"bus {cat}"] += 1

        dists, use_shape = along_route_km(r.geometry, lats, lons)
        if not use_shape:
            feed.log[ck]["bus shape omitted: line passes a stop twice"] += 1
        offsets = [0.0]
        for i in range(1, len(dists)):
            seg = dists[i] - dists[i - 1]
            if ck == "sh":
                mid = Point((lons[i] + lons[i - 1]) / 2, (lats[i] + lats[i - 1]) / 2)
                v = SH_SPEED_INSIDE_RING_KMH if ring.contains(mid) else SH_SPEED_OUTSIDE_RING_KMH
            else:
                v = OTHER_BUS_SPEED_KMH
            offsets.append(offsets[-1] + seg / v * 3600)

        rid = f"{ck}_b_{r.name}"
        stop_ids = [f"{ck}_{s}" for s in g["stop_id"]]
        short, long_ = route_names(r)
        feed.routes.append({"route_id": rid, "agency_id": ck, "route_short_name": short,
                            "route_long_name": long_, "route_desc": cat,
                            "route_type": 4 if cat == "ferry" else 3})
        shape_id = rid if use_shape else ""
        if use_shape:
            feed.add_shape(rid, r.geometry)
        for svc, windows in services.items():
            if windows:
                feed.add_trip(rid, f"{rid}_{svc}", svc, shape_id, stop_ids, offsets, windows, headsign=short)


def build_metro(feed, ck, routes, seqs):
    kept = []
    for _, r in routes.iterrows():
        name = r["route_cn"]
        reason = metro_drop_reason(ck, name, r["status"])
        is_night_shift = ck == "sh" and "夜班" in name and str(r["status"]) == "1"
        if reason and not is_night_shift:
            feed.log[ck][f"metro dropped: {reason}"] += 1
            continue
        g = seqs.get(name)
        if g is None or len(g) < 2:
            feed.log[ck]["metro dropped: no stops"] += 1
            continue
        key = metro_line_key(name)
        tram = key == "tram"
        lats, lons = g.geometry.y.tolist(), g.geometry.x.tolist()
        dists, use_shape = along_route_km(r.geometry, lats, lons)
        if not use_shape:
            feed.log[ck]["metro shape omitted: line passes a stop twice"] += 1
        v = TRAM_SPEED_KMH if tram else METRO_SPEED_KMH
        offsets = [d / v * 3600 for d in dists]
        rid = f"{ck}_m_{r.name}"
        stop_ids = [f"{ck}_{s}" for s in g["stop_id"]]
        short, long_ = route_names(r)
        feed.routes.append({"route_id": rid, "agency_id": ck, "route_short_name": short,
                            "route_long_name": long_, "route_desc": "tram" if tram else "metro",
                            "route_type": 0 if tram else 1})
        shape_id = rid if use_shape else ""
        if use_shape:
            feed.add_shape(rid, r.geometry)
        if is_night_shift:
            # Two late trains from Hongqiao, Sunday-Thursday (Wikipedia). Exact
            # departure times are not published there: 23:15 and 23:45 used.
            for i, dep in enumerate((1395, 1425)):
                feed.add_trip(rid, f"{rid}_SUNTHU_{i}", "SUNTHU", shape_id, stop_ids, offsets,
                              start_s=dep * 60, headsign=name)
            feed.log[ck]["metro night-shift trains (2/night Sun-Thu)"] += 1
            continue
        kept.append((r, rid, stop_ids, offsets, key, shape_id))
        feed.log[ck]["metro routes"] += 1

    # Branched Shanghai lines: group same-line routes by direction.
    by_line = defaultdict(list)
    for item in kept:
        by_line[item[4]].append(item)
    for key, items in by_line.items():
        groups = direction_groups(items) if ck == "sh" and key in SH_METRO_OUTER else [[it] for it in items]
        for grp in groups:
            if len(grp) == 1:
                r, rid, stop_ids, offsets, _, shp = grp[0]
                profile, hours = metro_profile(ck, key)
                add_metro_services(feed, ck, key, rid, shp, stop_ids, offsets, profile, hours, route_names(r)[0])
                if (ck, key) in SHORT_TURNS:
                    a, b, st_profile = SHORT_TURNS[(ck, key)]
                    names = seqs[r["route_cn"]]["name_cn"].tolist()
                    i, j = sorted((names.index(a), names.index(b)))
                    add_metro_services(feed, ck, key, rid, shp, stop_ids[i:j + 1],
                                       [o - offsets[i] for o in offsets[i:j + 1]], st_profile, hours,
                                       route_names(r)[0] + f" (short-turn {a}-{b})", trip_suffix="_short")
                    feed.log[ck][f"metro line {key}: short-turn {a}-{b} ({j - i + 1} stops)"] += 1
                continue
            inner, hours = metro_profile(ck, key)
            outer = SH_METRO_OUTER[key]
            for r, rid, stop_ids, offsets, _, shp in grp:
                add_metro_services(feed, ck, key, rid, shp, stop_ids, offsets, outer, hours, route_names(r)[0])
            r0, rid0, stop_ids0, offsets0, _, shp0 = grp[0]
            common = set.intersection(*(set(it[2]) for it in grp))
            idx = [i for i, s in enumerate(stop_ids0) if s in common]
            trunk = {b: (1 / (1 / inner[b] - len(grp) / outer[b]) if 1 / inner[b] > len(grp) / outer[b] else None)
                     for b in inner}
            t_ids = [stop_ids0[i] for i in idx]
            t_off = [offsets0[i] - offsets0[idx[0]] for i in idx]
            add_metro_services(feed, ck, key, rid0, shp0, t_ids, t_off, trunk, hours, route_names(r0)[0] + " (short-turn)",
                               trip_suffix="_trunk")
            feed.log[ck][f"metro line {key}: {len(grp)} branches + trunk short-turn ({len(t_ids)} stops)"] += 1


def direction_groups(items):
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][2], items[j][2]
            common = [s for s in a if s in set(b)]
            if len(common) >= 2 and [b.index(s) for s in common] == sorted(b.index(s) for s in common):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[find(i)].append(it)
    return list(groups.values())


def metro_profile(ck, key):
    if ck == "sh":
        return SH_METRO[key], SH_METRO_HOURS
    table = OTHER_METRO[ck]
    profile = table.get(key) or (table.get("tram") if key == "tram" else None) or table["default"]
    return profile, OTHER_METRO_HOURS.get(ck, OTHER_METRO_HOURS["default"])


def add_metro_services(feed, ck, key, rid, shape_id, stop_ids, offsets, profile, hours, headsign, trip_suffix=""):
    open_m, close_m = hours
    for svc, bands in (("WD", WD_BANDS), ("WE", WE_BANDS)):
        w = banded(profile, open_m, close_m, bands)
        if w:
            feed.add_trip(rid, f"{rid}{trip_suffix}_{svc}", svc, shape_id, stop_ids, offsets, w, headsign=headsign)
    if ck == "sh" and key in SH_METRO_FRISAT_EXTENDED and profile.get("late"):
        feed.add_trip(rid, f"{rid}{trip_suffix}_FRISAT", "FRISAT", shape_id, stop_ids, offsets,
                      [(close_m, close_m + 60, profile["late"])], headsign=headsign)


def build_maglev(feed):
    d = json.load(open(os.path.join(SOURCES, "osm_overpass_maglev.json")))
    lines = [LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
             for e in d["elements"] if e["type"] == "way" and len(e.get("geometry", [])) >= 2]
    merged = linemerge(lines)
    line = max(getattr(merged, "geoms", [merged]), key=lambda g: g.length)
    (lon0, lat0), (lon1, lat1) = line.coords[0], line.coords[-1]
    if lon0 > lon1:  # orient Longyang Road (west) -> Pudong Airport (east)
        line = LineString(list(line.coords)[::-1])
        (lon0, lat0), (lon1, lat1) = line.coords[0], line.coords[-1]
    length_km = along_line_km(line, [lat0, lat1], [lon0, lon1])[-1]
    feed.stops["sh_maglev_longyang"] = {"stop_id": "sh_maglev_longyang", "stop_name": "Longyang Road (Maglev)",
                                        "stop_lat": round(lat0, 7), "stop_lon": round(lon0, 7)}
    feed.stops["sh_maglev_pvg"] = {"stop_id": "sh_maglev_pvg", "stop_name": "Pudong International Airport (Maglev)",
                                   "stop_lat": round(lat1, 7), "stop_lon": round(lon1, 7)}
    feed.routes.append({"route_id": "sh_maglev", "agency_id": "sh", "route_short_name": "Maglev",
                        "route_long_name": "Shanghai Maglev (Longyang Road - Pudong International Airport) / 上海磁浮示范运营线",
                        "route_desc": "maglev, 300 km/h cruise", "route_type": 2})
    for direction, stop_ids, geom in (("out", ["sh_maglev_longyang", "sh_maglev_pvg"], line),
                                      ("back", ["sh_maglev_pvg", "sh_maglev_longyang"],
                                       LineString(list(line.coords)[::-1]))):
        shape_id = f"sh_maglev_{direction}"
        feed.add_shape(shape_id, geom)
        for svc in ("WD", "WE"):
            feed.add_trip("sh_maglev", f"sh_maglev_{direction}_{svc}", svc, shape_id, stop_ids,
                          [0, MAGLEV_RUN_SECONDS], MAGLEV_WINDOWS, headsign="Maglev")
    feed.log["sh"][f"maglev: OSM line {length_km:.1f} km, {MAGLEV_RUN_SECONDS}s run"] += 1


ring_poly = None


def main():
    global ring_poly
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simplify-deg", type=float, default=0.00002,
                    help="Shape simplification tolerance in degrees (~2 m); 0 keeps every CPTOND vertex")
    args = ap.parse_args()

    ring_poly = gpd.read_file(os.path.join(SOURCES, "outer_ring_s20_polygon.geojson")).geometry.iloc[0]
    ring = prep(ring_poly)
    load_zhwiki()
    feed = Feed(args.simplify_deg)
    for city_dir, prefix, ck, agency in CITIES:
        build_city(feed, city_dir, prefix, ck, agency, ring)
        print(f"[{ck}] routes so far={len(feed.routes)} trips={len(feed.trips)}", flush=True)
    build_maglev(feed)

    used = {st["stop_id"] for st in feed.stop_times}
    stops = [s for sid, s in feed.stops.items() if sid in used]
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "agency.txt"), feed.agencies,
              ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang"])
    write_csv(os.path.join(args.out, "calendar.txt"),
              [{"service_id": sid, **dict(zip(["monday", "tuesday", "wednesday", "thursday", "friday",
                                               "saturday", "sunday"], days)),
                "start_date": CALENDAR_START, "end_date": CALENDAR_END} for sid, days in SERVICES.items()],
              ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
               "start_date", "end_date"])
    write_csv(os.path.join(args.out, "stops.txt"), stops, ["stop_id", "stop_name", "stop_lat", "stop_lon"])
    write_csv(os.path.join(args.out, "routes.txt"), feed.routes,
              ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc", "route_type"])
    write_csv(os.path.join(args.out, "trips.txt"), feed.trips,
              ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"])
    write_csv(os.path.join(args.out, "stop_times.txt"), feed.stop_times,
              ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
    write_csv(os.path.join(args.out, "frequencies.txt"), feed.freqs,
              ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"])
    write_csv(os.path.join(args.out, "shapes.txt"), feed.shapes,
              ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])

    summary = {ck: dict(sorted(v.items())) for ck, v in feed.log.items()}
    summary["_totals"] = {"routes": len(feed.routes), "trips": len(feed.trips), "stops": len(stops),
                          "stop_times": len(feed.stop_times), "frequencies": len(feed.freqs),
                          "shape_points": len(feed.shapes)}
    with open(os.path.join(SOURCES, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
