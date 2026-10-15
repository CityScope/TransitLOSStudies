# How the Shanghai metropolitan-area GTFS feed was built

Feed: `shanghai/gtfs/shanghai_metro_area/` (built 2026-09-28)
Builder: `china/CPTOND-2025/code/build_shanghai_metro_area_gtfs.py`
Every source cited below is archived in this folder (`sources/`); the file
name is given in brackets, e.g. [sh_bus_service_spec_2024.html].

No official GTFS exists for any mainland Chinese city. The feed combines:

- a **real network**: routes, ordered stops, stop coordinates and route
  alignments from the CPTOND-2025 dataset;
- **researched service levels**: headways, first and last departures, peak
  periods and running speeds. These come from official standards,
  statistics, operator timetables and Wikipedia, and each value below names
  its source.

Where no source exists, the value is marked as an **assumption**.

Rebuild from `city_science_network/` (takes about 35 s):

```
uv run python china/CPTOND-2025/code/build_shanghai_metro_area_gtfs.py \
    --out shanghai/gtfs/shanghai_metro_area
```

The feed that was used until 2026-09-28 (Shanghai only, uniform bus
headways, 06:00–22:00 only) is kept in
`shanghai/sources/gtfs_backups/shanghai_cptond_until_20260928/`. It sits
outside `gtfs/` because the pipeline loads every folder in `gtfs/` as a feed.

---

## 1. Coverage

The feed covers the whole **"1+8" Shanghai Metropolitan Area** (上海大都市圈):
Shanghai plus Suzhou, Wuxi, Changzhou, Nantong, Jiaxing, Huzhou, Ningbo and
Zhoushan. It includes every route CPTOND has for those nine cities, not only
the study AOI, and adds the Shanghai Maglev, which is not in CPTOND.
Jiaxing, Huzhou and Zhoushan have no metro.

| | routes | trips | stops | stop_times | frequency rows | shape points |
|---|---|---|---|---|---|---|
| total | 13,662 | 27,148 | 76,111 | 679,676 | 87,313 | 1,213,435 |

Per-city counts by route category are in [build_summary.json], which the
builder rewrites on every run.

The pipeline keeps only the stops inside `aoi.gpkg`, about 16,900, almost
all in Shanghai. The map draws every route that touches the AOI along its
full length, so routes crossing into Kunshan or Jiaxing appear whole.

## 2. Base network: CPTOND-2025

- **Source:** China Public Transport Operation Network Dataset (Nature
  Scientific Data 2025, doi:10.6084/m9.figshare.29377427)
  [cptond2025_paper_nature_sdata.html]. Provenance and download notes are in
  `china/sources/sources.md`.
- **What it provides:** each route (one per direction), its ordered stops,
  stop coordinates, the route polyline, route type, and an operating status.
  About 45% of bus routes also have first and last departure times.
- **Checks:** CPTOND's Shanghai bus network adds up to 25,227 km of route,
  against the official 25,185 km [sh_transport_report_2022_sina.html]. Every
  stop lies exactly on its route line, and every line is drawn in stop order.
- **Mapping to GTFS:**
  - Each CPTOND route becomes one GTFS route. Each route has one trip per
    service pattern (weekday and weekend).
  - Stop times are relative. `frequencies.txt` (`exact_times=0`) sets the
    headway for each time band.
  - IDs are prefixed by city (`sh_`, `sz_`, `wx_`, `cz_`, `nt_`, `jx_`,
    `hz_`, `nb_`, `zs_`).
  - Names are English with the Chinese in `route_long_name`.
  - `route_desc` holds the service category used below.

### Removed or corrected

| What | Why | Count |
|---|---|---|
| Metro routes with `status` 2 or 3 | Not operating in CPTOND's snapshot: Line 22, the Jiamin line, and the Line 2/12/13/15/18 extensions in Shanghai; similar routes elsewhere | 18 in Shanghai, 20 in other cities |
| Bus routes with `status` 0 | Suspended | 452 |
| Suzhou's copy of **Shanghai Metro Line 11** | Duplicate of the Shanghai route | 2 |
| Express variants (Shanghai Line 16 大站车, Suzhou Line 11 大站车) | Already covered by the line headway | 4 |
| Bus routes whose stop sequence is identical to a route already in the feed | Cross-border routes listed by both cities | 169 |
| 城际铁路 "bus" routes | Intercity rail lines filed as bus in CPTOND | 8 |

Earlier versions of this feed kept all of these. The unopened lines were
getting full operating-line headways.

## 3. Calendars and time bands

**Calendars.** `WD` runs Monday–Friday and `WE` Saturday–Sunday. `FRISAT`
covers the metro's Friday/Saturday late extension, and `SUNTHU` the Line 2
and Line 10 late trains. All are valid 2025-01-01 to 2027-12-31. The study
uses a representative weekday, so it reads the `WD` services.

**Weekday time bands:**
- early: before 07:00
- **morning peak: 07:00–09:00**
- midday: 09:00–17:00
- **evening peak: 17:00–19:00**
- evening: 19:00–21:00
- late: after 21:00

Weekend bands are early / day (07:00–19:00) / evening / late. The peak
periods follow the peak columns of the Shanghai Metro table and the
07:00/07:30–09:00 and 16:30–18:30 peaks given for Changzhou, Nantong and
Wuxi.

Each route only runs within its own first-to-last span, and the bands are
cut to that span. Night lines are written on the previous evening's service
day (times past 24:00 are written as 24:xx–29:xx).

**Result.** The feed is time-aware. The table shows the number of Shanghai
routes running at each time on a weekday:

| | 02:00 | 05:00 | 08:00 | 12:00 | 18:00 | 21:30 | 23:30 |
|---|---|---|---|---|---|---|---|
| day buses | 0 | 221 | 3,091 | 3,048 | 2,986 | 1,023 | 131 |
| night buses | 80 | 9 | 0 | 0 | 0 | 2 | 40 |
| metro lines | 0 | 0 | 44 | 44 | 44 | 44 | 0 |
| maglev | 0 | 0 | 1 | 1 | 1 | 1 | 0 |

- **Peak vs off-peak:** day routes run at their peak headway 07:00–09:00
  and 17:00–19:00, and at their off-peak or late headway otherwise.
- **Late evening:** the day buses still running at 23:30 are real late last
  departures, for example 49路 runs until 23:50.
- **Middle of the night:** between about 01:00 and 04:30 only night routes
  run.

## 4. Buses

### 4.1 Shanghai: route classes

These follow the **Shanghai bus service standard**
(上海市公共汽车和电车客运服务规范, 2024, articles 22–23)
[sh_bus_service_spec_2024.html]:

- **Zones** are defined by the S20 Outer Ring (外环), built from OSM
  [osm_overpass_s20_outer_ring_ways.json → outer_ring_s20_polygon.geojson,
  655 km², matching the ~660 km² usually quoted].
  - *central*: both terminals inside the ring, or crossing it with at least
    two-thirds of the route length inside (the standard's own rule).
  - *radial*: one terminal inside.
  - *suburban*: both terminals outside.
- **Special classes** are detected from the route name, Shanghai's route
  numbering, the zh.wikipedia route list and CPTOND type:
  - **night**, which covers four cases:
    - names containing 夜宵/夜班/夜间/通宵;
    - **every 3xx route**: zh.wikipedia's list of numbered routes states
      "3字头三位数编号线路均为夜宵线，运行在每日的23时至次日6时之间", so
      301路–3xx路 are night routes even though their names don't say so;
    - routes the list marks 夜宵线;
    - any route whose first departure is 21:00 or later, for example 62路,
      listed as running 23:40–04:30;
  - **peak-only**, which covers two cases:
    - names containing 高峰 (早/上午 = morning only, 晚/下午 = evening only);
    - **2xx routes** (the list: "2开头的三位数线路：高峰线"), unless it lists
      them under 全日运营线路 (all-day routes), such as 220路.
    Peak-only routes run only inside 05:30–09:30 and 15:30–19:30, even when
    the list gives one continuous range (777路: 06:45–19:00).
  - trunk (71路 BRT and 中运量 medium-capacity lines)
  - customized (DZ / 定制)
  - fixed-schedule (定班/村村通/最后一公里)
  - community/micro/village (社区, 微巴, 微公交, 乡村, 便民, "Community car")
  - ferry, tourist, airport bus

### 4.2 Headways

The standard sets **maximum** headways:
- central numbered routes: ≤ 8 min at peak, ≤ 20 min otherwise;
- trunk routes: ≤ 5 min at peak, ≤ 15 min off-peak;
- peak-only routes: ≤ 8 min;
- night routes: ≤ 40 min (numbered) or ≤ 60 min (named);
- village, last-mile and fixed-schedule routes: ≤ 60 min;
- all other routes: ≤ 25 min.

**Fleet calibration.** The fleet size limits what the network can run:
- **Fleet:** about 17,000 buses. The official 2024 report gives 2,999,000
  riders per day and 176.9 riders per bus per day, which implies ~16,950
  buses [sh_transport_annual_2024_urban_passenger_CAPTURE.md]. The 2021
  fleet was 17,631 [sh_transport_report_2022_sina.html].
- **Buses needed** to run a set of headways = Σ (one-way running time ×
  1.15 layover ÷ headway), using CPTOND route lengths and the speeds in
  §4.4. The 1.15 layover factor is an assumption.

Peak headway on every route, and the buses that would need:

| Peak headway: central / radial / suburban | Buses needed at peak |
|---|---|
| 4 / 4 / 4 (previous feed) | ~43,500–51,000 |
| 6 / 8 / 10 | ~21,600 |
| 7 / 10 / 12 | ~18,100 |
| **8 / 12 / 15 (used)** | **~15,200** |
| 8 / 12 / 18 | ~14,200 |

The chosen values need about 90% of the fleet in service at peak. The
finished feed runs **14,609 buses at 08:00 on a weekday**, about 86% of the
fleet: night routes and peak-only routes no longer run all day. Central routes
land exactly at the standard's 8‑min ceiling, so in practice the fleet, not
the standard, is what binds.

Individual routes are consistent with this:
- Route 55, a trunk-like central route: 3–5 min peak, 8–12 min off-peak
  [wikipedia_shanghai_bus_route_55.html].
- Trolleybus routes 14 and 20: 2–3 min peak, 5–15 min off-peak.
- Route 817: 8 min peak / 20 min off-peak. Pudong route 108: 15 / 20.

**Shanghai bus headways (minutes):**

| Class | early | am peak | midday | pm peak | evening | late | weekend day | Basis |
|---|---|---|---|---|---|---|---|---|
| central | 12 | 8 | 12 | 8 | 12 | 20 | 12 | standard ceiling + fleet calibration |
| radial | 15 | 12 | 15 | 12 | 15 | 25 | 15 | fleet calibration |
| suburban | 20 | 15 | 20 | 15 | 20 | 25 | 20 | fleet calibration; ≤ 25 ceiling |
| trunk (71路, 中运量) | 10 | 5 | 10 | 5 | 10 | 15 | 10 | standard (≤ 5 / ≤ 15) |
| peak-only | – | 8 (06:00–08:30) | – | 8 (16:30–19:00) | – | – | – | standard art. 22–23; weekdays only |
| night (overnight) | the published headway for 66 routes (mostly 30–40 min, e.g. 301路 30, 302路 35); otherwise 40 (numbered) or 60 (named); every day ||||||| zh.wikipedia route list; standard ceilings |
| fixed-schedule | 60 all day ||||||| standard ceiling ("挂牌" service) |
| community / micro / village | 30 all day ||||||| **assumption** |
| customized (DZ, 定制) | weekdays, 07–09 and 17–19 only, 20 min ||||||| **assumption** (the DZ418 timetable, previously copied to every DZ route, is not typical: DZ301 runs weekly) |
| ferry | 20 | 15 | 20 | 15 | 20 | 25 | 20 | Shanghai Ferry lines run every 10–30 min [wikipedia_shanghai_ferry.html] |
| tourist, airport bus | 20 all day ||||||| **assumption** within the 15–45 min range of published airport-bus intervals |

- **Off-peak values** (12 / 15 / 20) sit below the standard's ≤ 20 / ≤ 25
  ceilings. They follow the ratio between peak and off-peak headways in the
  route examples above; this ratio is an **assumption**.
- **Weekend buses** use the midday headway all day. **Assumption:** no
  weekend bus data was found.

### 4.3 Service span (first and last departures)

First and last departures are taken from the first available source below.

1. **zh.wikipedia list of numbered routes (Shanghai):** per-direction first
   and last departures for 820 day routes, 70 night routes and 20 peak-only
   routes [zhwiki_shanghai_bus_routes_numbered.html, parsed by
   `parse_zhwiki_shanghai_routes.py` into
   zhwiki_shanghai_numbered_routes_parsed.csv]. The direction is matched by
   comparing CPTOND's origin terminal with the list's left/right terminal.
2. **CPTOND times** (~45% of routes).
3. **Shanghai routes with neither** use the standard's minimum span
   (art. 22):
  - central numbered routes: 05:30–22:30;
  - central named routes and radial routes: 06:00–21:00;
  - suburban routes: 06:00–19:00;
  - peak-only routes: 06:00–08:30 and 16:30–19:00;
  - night routes: 23:00–05:00.
- **Other cities' routes without times** use the median first and last
  departure of that city's own CPTOND routes.
- **Other cities' 夜班/夜间 routes** are evening services, not overnight
  ones: those with CPTOND times run about 18:00–21:30. They are labelled
  `evening`. Those without times use their city's median evening span, for
  example Suzhou 18:30–20:40 and Ningbo 18:40–22:00. Only names containing
  夜宵/通宵 get the overnight 23:00–05:00 default outside Shanghai.

### 4.4 Running speed

- **Shanghai:** each stop-to-stop segment is timed at **16 km/h inside the
  Outer Ring** and **21 km/h outside**. The official 2021 peak averages
  were 16.1 / 15.8 km/h in the central city and 19.6 / 19.2 km/h citywide
  [sh_transport_report_2022_sina.html]. With these zone speeds the network
  average is 19.2 km/h.
- **Other cities:** 20 km/h. **Assumption:** no citywide speed was found.
- Distances are measured **along the real route line**, not in a straight
  line, so the speed pyGTFSHandler computes from `shapes.txt` equals the
  speed used here.
- For routes whose shape is omitted (§7), stop-to-stop straight-line
  distance is used instead.

### 4.5 Other cities' buses

Regular routes run **20 min at peak, 25 off-peak and 30 late**. Other route
classes use the same rules as Shanghai.

- **Basis:** Suzhou's 2024 statistical bulletin gives 5,316 buses on 791
  urban routes [metro_area/bulletin_2024_suzhou.html]. At CPTOND's Suzhou
  mean route length of 17.5 km, 20 km/h, the same layover and 90% of buses
  in service, that is about 20 min at peak.
- Wuxi's bulletin gives 343 routes and 6,236 km, but no fleet count
  [metro_area/bulletin_2024_wuxi.html].
- **Caveat:** these values are transferred from Suzhou, and they are the
  least certain part of the feed. They mostly affect areas outside the
  study AOI.

## 5. Metro

### 5.1 Shanghai Metro

**Headways** come from Wikipedia's "Shanghai Metro" headway table, last
updated July 2023 and sourced there to the operator
[wikipedia_shanghai_metro.wikitext, section "Operating hours and train
intervals"]. The feed uses its inner-section values: morning peak, mid-day,
evening peak and weekend daytime, plus the "other hours" range. For other
hours:
- the lower bound of the range applies 19:00–21:00;
- the upper bound applies after 21:00;
- the midpoint applies before 07:00.

The Pujiang line uses the table's single (outer) column.

| Line | am peak | midday | pm peak | weekend | other hours |
|---|---|---|---|---|---|
| 1 | 2'30 | 4'00 | 3'00 | 4'00 | 4'00–12'00 |
| 2 | 2'30 | 4'00 | 3'00 | 3'40 | 5'00–11'00 |
| 3 | 5'00 | 7'00 | 5'00 | 4'00 | 6'30–10'00 |
| 4 | 3'10 | 8'00 | 5'00 | 6'30 | 5'00–12'00 |
| 5 | 2'30 | 4'30 | 3'00 | 4'00 | 7'30–10'00 |
| 6 | 2'00 | 4'30 | 2'30 | 3'20 | 5'00–10'00 |
| 7 | 1'55 | 5'30 | 2'44 | 5'00 | 5'00–8'00 |
| 8 | 2'00 | 4'20 | 2'45 | 3'40 | 5'00–6'00 |
| 9 | 1'50 | 5'30 | 2'30 | 5'00 | 6'00–10'00 |
| 10 | 2'30 | 4'00 | 3'00 | 3'45 | 5'00–8'00 |
| 11 | 2'00 | 6'00 | 3'00 | 5'00 | 6'00–8'00 |
| 12 | 2'30 | 6'00 | 3'45 | 6'00 | 6'00–10'00 |
| 13 | 2'30 | 6'00 | 3'00 | 5'00 | 6'00–10'00 |
| 14 | 3'20 | 6'00 | 4'00 | 5'00 | 5'00–12'00 |
| 15 | 3'40 | 8'00 | 4'30 | 6'00 | 5'00–13'00 |
| 16 | 3'30 | 7'00 | 4'00 | 5'00 | 5'00–10'00 |
| 17 | 3'30 | 10'00 | 4'00 | 7'00 | 8'00–10'00 |
| 18 | 3'00 | 7'00 | 3'30 | 6'30 | 6'30–12'00 |
| Pujiang | 3'20 | 6'30 | 3'50 | 6'00 | 9'00 |

**Branched lines 5, 10 and 11.** CPTOND lists each branch as a separate full
route. Giving every branch the line's full headway would count the shared
trunk twice. Instead:
- Each branch gets the table's **outer-section** (maximum-interval) values:

  | Line | am peak | midday | pm peak | weekend | other hours |
  |---|---|---|---|---|---|
  | 5 | 7'30 | 9'00 | 9'00 | 8'00 | 7'30–10'00 |
  | 10 | 9'00 | 12'00 | 9'00 | 10'00 | 6'00–14'00 |
  | 11 | 10'00 | 12'00 | 7'00 | 10'00 | 11'00–18'00 |

- An extra **short-turn service** runs over the stations shared by all
  branches: 7 for Line 5 (Xinzhuang–Dongchuan Road), and 30 each for Lines 10
  and 11. Its headway is set so that the combined frequency on the trunk
  equals the inner-section value: 1/h_short = 1/h_inner − 2/h_outer.

**Other Shanghai Metro services:**
- **Hours:** 05:30–23:00. Wikipedia says most stations open 5:00–6:00 and
  close 22:30–23:00.
- **Friday/Saturday extension:** one extra hour on Lines 1, 2, 7, 9, 10 and
  13, at the late headway (Wikipedia, "Extended hours on Friday and
  Saturday").
- **Line 2 / Line 10 night-shift (夜班) routes:** two late trains from
  Hongqiao on Sunday–Thursday nights, stopping at selected stations
  (Wikipedia). They are modelled as timetabled trips at 23:15 and 23:45.
  **The exact times are an assumption.**
- **Line 4:** its inner and outer circles are the two directions, each at
  the line headway.
- **Speed:** 35 km/h along the track for all lines. **Assumption:** most
  lines have an 80 km/h maximum, and Lines 11–17 have 100 km/h (Line 16,
  120 km/h).

### 5.2 Other cities' metro

All values are in minutes (m'ss = minutes and seconds).

| City / line | Peak | Off-peak | Late | Weekend | Source (in `metro_area/`) |
|---|---|---|---|---|---|
| Suzhou L1 | 3 (am) / 4 (pm) | 8 | 10* | 6'50 | suzhou_metro_schedule_adjust_163.html (Apr 2022 notice, during COVID service cuts; the 2018 timetable in zhwiki_suzhou_line1.html shows ≤ 4 min off-peak, so 8 is conservative) |
| Suzhou L2 | 4'30 (am) / 5'20 (pm) | 6'45 | 10 | 5'20 | zhwiki_suzhou_line2.html |
| Suzhou L4 | 4'30 (am) / 6 (pm) | 8 | 10* | 8 | suzhou_metro_schedule_adjust_163.html (2022) |
| Suzhou L7 | 6 | 7 | 10* | 7 | suzhou_metro_line7_thepaper.html |
| Suzhou L11 (Kunshan) | 6'30 (6–7) | 8 | 10* | 8 | zhwiki_suzhou_line11.html |
| Suzhou L3, L5, L6, L8 | 6* | 8 | 10* | 8 | 2022 notice gives 8 all day for L3/L5; peak **assumed** |
| Suzhou trams, Wujiang ART | 10* | 12* | 15* | 12* | **assumption** |
| Wuxi L1, L2 | 6 | 8 | 9 (after 21:00) | 8 | wuxi_metro_line2_interval_bendibao.html (2020 notice for L1 and L2) |
| Wuxi L3, L4, S1 | 6* | 8* | 9* | 8* | **assumed** equal to L1/L2 |
| Ningbo L1 | 3'50 | 7'40 full line, plus short-turn (below) | 7'40 | 5'30 | zhwiki_ningbo_line1.html |
| Ningbo L2 | 4'20 | 4'54 | 10 | 6'20 | zhwiki_ningbo_line2.html |
| Ningbo L3 | 9 full line, plus short-turn | 9, plus short-turn | 10 | 6'30 | zhwiki_ningbo_line3.html |
| Ningbo L4 | 4'25 | 6'20 | 10 | 6'25 | zhwiki_ningbo_line4.html |
| Ningbo L5 | 4'58 | 5'52 | 7'40 | 5'52 | zhwiki_ningbo_line5.html |
| Changzhou L1 (L2 assumed the same) | 7 | 8 | 10 | 7 | changzhou_metro_line1_timetable_bendibao.html |
| Nantong L1 (L2 assumed the same) | 7 | 8 | 14'50 | 8 | nantong_metro_line1_bendibao.html; zhwiki_nantong_line1.html |

\* = no published figure; assumed.

**Ningbo short-turn services** (小交路) run on top of the full-line trains:
- Line 1, 望春桥–东环南路: 7'40 off-peak, giving 3'50 on the core.
- Line 3, 大通桥–明辉路: 9' until 20:00, giving 4'30 on the core.

**Hours:**
- Changzhou and Nantong: 06:00–22:00 (sourced).
- Ningbo: 05:45–22:00 (Line 5 page).
- Suzhou: 05:40–22:30 (Line 11 page).
- Others: 06:00–22:30. **Assumption.**

Speed is 35 km/h for metro lines and 18 km/h for trams/ART (both
**assumptions**).

## 6. Shanghai Maglev

This is not in CPTOND.
- **Alignment:** built from OSM ("上海磁浮示范运营线") [osm_overpass_maglev.json].
  The 29.9 km line is merged, and stops are placed at its ends: Longyang
  Road and Pudong International Airport.
- **Timetable:** 8 min 10 s per trip, an average of 224 km/h at the 300 km/h
  cruise speed used since May 2021. It runs every day 06:45–21:42, **every
  20 min**, with 15 min between the first two departures
  [wikipedia_Shanghai_maglev_train.wikitext; maglev_chinaholiday_timetable.html;
  maglev_shanghai_airport_com.html].
- **Route type:** `route_type` 2 (rail), so transitLOS scores it as rail.

## 7. Shapes (`shapes.txt`)

**Where they come from:** each route's CPTOND polyline is written as its
shape, simplified to about 2 m, so the map draws real alignments.

**Why some are omitted:** pyGTFSHandler places each stop on the shape at its
nearest point, one stop at a time, and ignores stop order and any provided
`shape_dist_traveled`. When a line passes the same stop more than once
(loops, out-and-back sections, circle lines), a stop can be placed on the
wrong pass. That produces absurd speeds, up to 900 km/h in testing.

**What the builder does:**
- It detects routes where any stop has a second pass of the line within
  about 10 m.
- It writes those routes **without a shape**, and their running times use
  stop-to-stop distance.
- That is **1,550 routes (11%)**, including Shanghai Metro Line 4's two
  circles. The map draws them stop to stop.

**Result:** only 23 of 16,501 bus stops in the AOI still compute above
40 km/h.

**Proper fix:** make pyGTFSHandler place stops in order along the shape;
then every route could keep its shape. This is not done, because it would
change pyGTFSHandler for every city.

## 8. Validation

**Feed integrity:** every trip has stop times; every frequency row, shape
reference and stop reference resolves; no trip ID is duplicated.

**Loaded through the pipeline's own stop preparation**
(`transitlos.stops.download_and_prepare_stops`, AOI filter, 06:00–22:00,
150 m station grouping), on a scratch copy:

| | Previous feed | This feed |
|---|---|---|
| Bus-station headway, median | 2.3 min | 8.7 min |
| Bus-station speed, median | 18.0 km/h | 20.9 km/h |
| Metro-station headway, median | 1.0 min | 1.7 min |
| Metro-station speed, median | 19.7 km/h | 34.7 km/h |
| Maglev (airport station) | – | 219.6 km/h |
| Buses implied at 08:00, Shanghai | ~43,500–51,000 | 14,609 (fleet ~17,000) |

Headways at stations are low because pyGTFSHandler combines every route at
each station (stops within 150 m) into a single frequency. This is the same
for every city in the study.

**Known side effect:** the maglev's Longyang Road stop is grouped with
nearby bus stops, so the station reports the combined bus + maglev headway
and a bus-level speed. The same happens at any mixed-mode station (see the
`station_modes` note in `transitlos/stops.py`).

## 9. How much to trust each part

| Part | Status |
|---|---|
| Route existence, stop order, stop positions, alignments | CPTOND data (verified against official network length) |
| Which lines operate (status filter, duplicates, variants) | CPTOND status plus the fixes in §2 |
| Shanghai Metro headways by line and time band | Sourced (Wikipedia table, July 2023, from the operator) |
| Shanghai Metro branch/trunk split | Sourced values; splitting method is modelled |
| Maglev | Sourced |
| Ningbo, Changzhou, Nantong, Suzhou L1/2/4/7/11, Wuxi L1/2 metro | Sourced (operator notices and Chinese Wikipedia, 2020–2023) |
| Other metro lines, trams | Assumed (city defaults) |
| Shanghai bus headways | Official ceilings plus fleet calibration; not measured per route |
| Other cities' bus headways | Calibrated on Suzhou's fleet only; least certain |
| Bus service spans | Shanghai: zh.wikipedia timetable for ~900 numbered routes, then CPTOND times, then the standard's minimums; other cities: CPTOND times (~45%), else the city median |
| Which routes are night, peak-only or all-day | Shanghai numbering convention (2xx, 3xx) and the zh.wikipedia list; route names elsewhere |
| Bus running speeds | Shanghai: official zone averages; other cities: assumed |
| Metro and tram running speed | Assumed (35 / 18 km/h) |
| Weekend bus service, customized/community/village routes | Assumed |

## 10. Source index

| File | Supports |
|---|---|
| sh_bus_service_spec_2024.html | Shanghai bus standard: headway ceilings (art. 23), first/last bus and night/peak-only hours (art. 22), central-route definition |
| sh_transport_report_2022_sina.html | 2021 fleet 17,631, 1,596 routes, 25,185 km, peak speeds 19.6/19.2 citywide and 16.1/15.8 central |
| sh_transport_annual_2024_urban_passenger_CAPTURE.md | 2024 riders and riders per bus → ~16,950 buses (text capture; the page is a JavaScript shell) |
| thepaper_bus_speed_1468502.html, sh_public_transport_12th_fyp.html, sh_transit_metropolis_2021_23.pdf | Background on bus speeds and targets (12–15 km/h average, 15–20 trunk) |
| wikipedia_shanghai_bus_route_55.html, wikipedia_shanghai_trolleybus_route_14.html, wikipedia_shanghai_trolleybus_route_20.html, wikipedia_shanghai_bus_route_DZ418.html | Individual route headways used as cross-checks |
| wikipedia_shanghai_ferry.html | Ferry intervals |
| qq_night_bus_feature_2025.html, thepaper_night_bus_hubs_30032712.html, xinmin_night_bus_20250327.html | Night-line hours (23:00–05:00) and hub night routes 316/320 |
| zhwiki_shanghai_bus_routes_numbered.html → zhwiki_shanghai_numbered_routes_parsed.csv | Numbering convention (2xx peak-only, 3xx night), per-direction first/last times, night-route headways |
| wikipedia_shanghai_metro.wikitext / .html | Metro headway table (inner and outer), hours, Fri/Sat extension, Hongqiao late trains |
| travelchinaguide_pujiang_line.html | Earlier Pujiang source (superseded by the Wikipedia table) |
| wikipedia_Shanghai_maglev_train.wikitext, maglev_chinaholiday_timetable.html, maglev_shanghai_airport_com.html, travelchinaguide_shanghai_getting_around.html | Maglev hours, interval, run time, speed |
| osm_overpass_maglev.json | Maglev alignment |
| osm_overpass_s20_outer_ring_ways.json, outer_ring_s20_polygon.geojson | Outer Ring zone boundary |
| cptond2025_paper_nature_sdata.html | CPTOND dataset description |
| metro_area/bulletin_2024_suzhou.html, metro_area/bulletin_2024_wuxi.html | Other-city bus fleets and networks |
| metro_area/*.html, metro_area/wikipedia_*.wikitext | Other-city metro headways and hours (table in §5.2) |
| build_summary.json | Per-city counts written by the builder |
