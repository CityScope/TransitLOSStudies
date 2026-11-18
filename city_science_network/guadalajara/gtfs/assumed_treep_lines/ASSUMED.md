# ASSUMED feed: Guadalajara lines from treep.app

Not official data. Lines of the Mi Transporte network (treep.app, agency 35) that are not in the official feed (mdb-2366, calendar 2024) or in OSM: C52, C87, C84, C121A, C81, T22-V02, C12, C58, C111-2, C51-V1, C96-V1, T06-2, T16B-C04, T16B-C05, C72.
STOPS: treep.app prints the stops of every trip (https://treep.app/index/trip/35/<route>/<trip>) and each stop page (https://treep.app/stops/35/<id>) carries its latitude/longitude; C72's stop ids encode them (jalisco_3_<lat>_W<lon>). These coordinates are the stops of the feed (stops closer than 150 m are merged); lines whose trip page lists no stops were placed from named stops geocoded on OSM.
PATH: shortest path on the OSM road network through the stops in order; an anchor that makes the path double back (more than 2.2x the straight hop + 800 m) is dropped. A line is written only if the implied speed (path / printed run time) is 10-40 km/h; loops (ends within 1.5 km) are exempt from the path-vs-straight-line check.
TIMES: the weekday timetable printed on the trip page: first departure, last departure, median gap between departures (the interval) and the median duration (run time); outbound and inbound are separate trips.
NOT DUPLICATES: each line was compared with the official feed (share of its stops within 60 m of the stops of the best-matching official route: 33-55% for C12, C58, C111-2, C96-V1, T06-2, T16B-C04, T16B-C05; C51-V1 69% but the official feed has V2 only).
Checked: stops are 12-17 m (median) from the drawn path on every line; 4 stops of 600 are 150-370 m off (C84, T22-V02).

HEADWAY_BASIS: line-specific published timetables, not feed medians.
