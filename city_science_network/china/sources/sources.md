China Public Transport Operation Network Dataset (CPTOND-2025)

National-scale bus + metro network dataset covering bus networks in 350
cities and metro systems in 46 cities across mainland China, Hong Kong,
Macao, and Taiwan.

Paper: https://www.nature.com/articles/s41597-025-06505-4
       (open-access mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC12886901/)
Data:  https://doi.org/10.6084/m9.figshare.29377427
Direct download: https://ndownloader.figshare.com/files/58153174 (CPTOND-2025.zip, ~2.16 GB)

Format: Shapefile (.shp), not native GTFS. No GTFS feed exists for
mainland Chinese cities (incl. Shanghai) in the MobilityData catalog or
elsewhere as of 2026-08. This is the closest available open dataset for a
Shanghai/China transit network.

Downloaded 2026-08-10: CPTOND-2025.zip (2.16 GB, md5
8b786408d8a9d4d16faa531f98cc9fc6, verified) and extracted to
./CPTOND-2025/ (13 GB uncompressed). Layout:
  CPTOND-2025/dataset/bus/shapefiles/<City>/<city>_bus_{routes,segments,stops,stops_unique}.shp
  CPTOND-2025/dataset/metro/shapefiles/<City>/<city>_metro_{routes,segments,stops,stops_unique}.shp
  CPTOND-2025/code/README.md has the field dictionary and processing scripts.
350 cities have bus shapefiles, 46 have metro shapefiles.

The Shanghai subset (bus + metro shapefiles) was copied out to
../shanghai/gtfs/bus_shapefiles/ and ../shanghai/gtfs/metro_shapefiles/.

--- 2026-08-10 GTFS conversion ---
Wrote a generic converter, CPTOND-2025/code/build_gtfs_from_cptond.py, that
turns any city's <city>_{bus,metro}_stops.shp + _stops_unique.shp into a
GTFS feed (works for any of the 350/46 cities, not just Shanghai -- pass
--city <pinyin> --data-root <dir with bus_shapefiles/ and/or
metro_shapefiles/>). Since CPTOND carries no timetable (no departure times,
no headway), the script builds a frequency-based feed: one representative
trip per route, running times from great-circle distance between
consecutive stops at an assumed speed (18 km/h bus / 35 km/h metro,
configurable), and a GTFS frequencies.txt row per trip with a configurable
assumed headway (12 min bus / 6 min metro by default) over a configurable
service window (06:00-22:00 by default), running daily (no calendar info
in the source). Ran for Shanghai:
../shanghai/gtfs/shanghai_cptond/ (3225 bus + 66 metro routes, 17506 stops,
69548 stop_times). Loads successfully via pyGTFSHandler.feed.Feed. Treat
stop placement/route topology as accurate (from the real network data);
treat exact running times as assumptions, not measurements.

--- 2026-08-10 follow-up: real headways for Shanghai Metro ---
Looked for real frequency data instead of a flat assumed headway. Found
Wikipedia's "Shanghai Metro" article, which has a per-line headway table
(morning-peak/midday/evening-peak/off-peak train intervals, sourced there
to the operator's official site) covering lines 1-18. Hardcoded that table
into build_gtfs_from_cptond.py (SHANGHAI_METRO_HEADWAY_MIN) and match each
CPTOND metro route to a line number parsed from its route_en name, emitting
5 frequency windows/day (peak 07:00-09:30 & 17:00-19:30, midday, other) at
the real headway for that line. 8 of 66 routes (the Jiamin and Pujiang
shuttle lines, plus Line 22 Phase II) aren't in that table and fall back to
a generic 5/8/10-min placeholder -- flagged in the script's stdout.

For bus, no bulk per-route frequency source exists (checked; a few
individual Shanghai bus-route Wikipedia pages have real headways, e.g.
peak 3-5 min / off-peak 8-12 min for high-frequency trunk routes, but
nothing covers the ~3000+ CPTOND routes) -- bus still uses a generic
peak/off-peak default (10/18 min), NOT measured per route. Treat only the
Shanghai Metro frequencies in this feed as real; bus frequencies (any city)
remain a rough placeholder.
