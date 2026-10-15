"""Manually-constructed GTFS feed for Andorra's national bus network (E.N.A.).

No official GTFS exists for Andorra. This feed is built from:

  - Govern d'Andorra's official published timetable leaflet ("El temps es
    teu", Agost 2026 edition), fetched 2026-08-10 from
    https://www.govern.ad/documents/d/guest/bus_linies_nacionals -- a 12-page
    PDF with one page per line (Le, L1-L7) giving the exact official
    departure-frequency schedule per direction and day-type (weekday /
    Saturday / Sunday-holiday), plus a per-line stop-sequence diagram with
    official numeric stop codes. This is the authoritative source for
    headways/operating windows used below (verbatim from the PDF, not
    estimated).
  - https://bus.ad/ line pages (fetched 2026-08-10), used to cross-check the
    PDF and confirm line topology (L1-L7); also clarified that the 8th line
    referenced in busos.ad's "Linies L1-L8" title is "Le" (a shuttle/express
    line between Fener and Pont de Fontaneda), not a missing "L8".
  - busos.ad (the realtime map) was checked for a scrapeable stops/timetable
    API -- it is a client-rendered Next.js app and no public API endpoint
    was found in its JS bundles, so it could not be used as a data source.

What is exact vs. approximate in this feed:
  - Headways and operating windows (GTFS frequencies.txt): EXACT, taken
    directly from the official PDF, per line/direction/day-type/window.
  - Stop identity for the ~11 stops used per line here: each maps to a real
    official stop code from the PDF diagrams (e.g. 5555 = Estacio Nacional,
    720 = Caldea, 624 = La Portalada). This feed uses only the terminals and
    major interchange stops per line (not the full ~20-30 stop official
    sequence per line) to keep the reconstruction tractable -- the full
    intermediate stop lists (with distinct codes per direction) are visible
    in the PDF but were not all transcribed here.
  - Stop coordinates: APPROXIMATE (town/landmark-centre coordinates, not the
    true stop-pole position), since the PDF gives stop codes/names but no
    coordinates.
  - In-vehicle running times between consecutive stops: derived from
    great-circle distance between those approximate coordinates at an
    assumed mountain-road average speed (18 km/h), NOT measured.
  - One-off "reinforcement" departures the PDF lists outside the regular
    headway pattern (e.g. a single extra 22:05 trip) are NOT included --
    only the repeating headway windows are encoded.
"""

from __future__ import annotations

import csv
import math
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gtfs", "ena_manual_reconstruction")

# ---------------------------------------------------------------------------
# Stops: official stop code -> (name, approx lat, approx lon).
# Codes/names are real (from the Govern d'Andorra PDF); coordinates are
# approximate town/landmark centres, not surveyed stop-pole positions.
# ---------------------------------------------------------------------------
STOPS = {
    "5555": ("Andorra la Vella - Estacio Nacional d'Autobusos", 42.5075, 1.5228),
    "720":  ("Escaldes-Engordany - Caldea", 42.5106, 1.5389),
    "624":  ("Sant Julia de Loria - La Portalada", 42.4639, 1.4913),
    "728":  ("Encamp - Central de FEDA", 42.5359, 1.5828),
    "241":  ("Canillo - Radio Andorra", 42.5676, 1.5975),
    "172":  ("Soldeu", 42.5769, 1.6693),
    "201":  ("Pas de la Casa - Parada de l'Esglesia", 42.5428, 1.7336),
    "416":  ("La Massana - Centre de la vila", 42.5449, 1.5147),
    "468":  ("Arinsal - Aparcament", 42.5747, 1.4839),
    "302":  ("Ordino - Portal del Riu (Auditori Nacional)", 42.5561, 1.5330),
    "341":  ("La Cortinada - Bordes del Riu", 42.5722, 1.5231),
    "5018": ("Fener", 42.5040, 1.5180),
    "616":  ("Pont de Fontaneda", 42.4750, 1.4970),
}

# ---------------------------------------------------------------------------
# Routes: ordered stop-code sequence (direction 0). Direction 1 is the
# reverse. Headway windows are per-direction and per day-type, transcribed
# verbatim from the official PDF (start, end, headway_minutes).
# ---------------------------------------------------------------------------
ROUTES = {
    "Le": {
        "name": "Fener - Pont de Fontaneda (Andorra la Vella - Sant Julia shuttle)",
        "stops": ["5018", "5555", "616"],
        "freq": {
            0: {"weekday": [("06:30", "07:00", 15), ("07:00", "10:00", 10),
                             ("10:00", "16:00", 20), ("16:00", "18:00", 10),
                             ("18:00", "22:00", 20)],
                "weekend": [("06:30", "22:00", 30)]},
            1: {"weekday": [("06:30", "07:00", 15), ("07:00", "10:00", 10),
                             ("10:00", "16:00", 20), ("16:00", "18:00", 10),
                             ("18:00", "22:00", 20)],
                "weekend": [("06:30", "22:00", 30)]},
        },
    },
    "L1": {
        "name": "Escaldes-Engordany - Sant Julia de Loria",
        "stops": ["720", "5555", "624"],
        "freq": {
            0: {"weekday": [("06:20", "21:00", 20), ("21:00", "22:00", 30)],
                "weekend": [("06:45", "22:15", 30)]},
            1: {"weekday": [("06:20", "21:00", 20), ("21:00", "22:00", 30)],
                "weekend": [("07:30", "22:30", 30)]},
        },
    },
    "L2": {
        "name": "Andorra la Vella - Encamp",
        "stops": ["5555", "720", "728"],
        "freq": {
            0: {"weekday": [("06:50", "21:35", 15)],
                "weekend": [("06:50", "21:20", 30)]},
            1: {"weekday": [("06:42", "21:27", 15)],
                "weekend": [("06:35", "21:35", 30)]},
        },
    },
    "L3": {
        "name": "Andorra la Vella - Soldeu",
        "stops": ["5555", "720", "728", "241", "172"],
        "freq": {
            # No Sunday service (per both bus.ad and the official PDF).
            0: {"weekday": [("06:40", "09:20", 20), ("14:50", "20:50", 20)],
                "saturday": [("06:40", "09:40", 30), ("16:10", "20:40", 30)]},
            1: {"weekday": [("06:50", "10:10", 20), ("15:20", "21:40", 20)],
                "saturday": [("06:50", "07:30", 20), ("07:30", "10:00", 30),
                             ("17:00", "21:30", 30)]},
        },
    },
    "L4": {
        "name": "Andorra la Vella - Pas de la Casa",
        "stops": ["5555", "720", "728", "241", "172", "201"],
        "freq": {
            0: {"weekday": [("06:25", "21:25", 30)],
                "weekend": [("06:25", "21:25", 30)]},
            1: {"weekday": [("06:40", "21:40", 30)],
                "weekend": [("06:40", "21:40", 30)]},
        },
    },
    "L5": {
        "name": "Andorra la Vella - Arinsal",
        "stops": ["5555", "416", "468"],
        "freq": {
            0: {"weekday": [("07:20", "08:20", 20), ("08:20", "22:20", 30)],
                "weekend": [("06:40", "21:40", 30)]},
            1: {"weekday": [("06:50", "08:50", 20), ("08:50", "22:20", 30)],
                "weekend": [("07:10", "21:40", 30)]},
        },
    },
    "L6": {
        "name": "Andorra la Vella - Ordino - La Cortinada",
        "stops": ["5555", "302", "341"],
        "freq": {
            0: {"weekday": [("06:55", "21:55", 30)],
                "weekend": [("06:45", "21:15", 30)]},
            1: {"weekday": [("07:40", "22:10", 30)],
                "weekend": [("07:00", "21:30", 30)]},
        },
    },
    "L7": {
        "name": "La Massana - Andorra la Vella - Ordino",
        "stops": ["416", "5555", "302"],
        "freq": {
            0: {"weekday": [("06:50", "22:10", 20)],
                "weekend": [("07:00", "22:00", 30)]},
            1: {"weekday": [("06:30", "21:50", 20)],
                "weekend": [("06:45", "21:45", 30)]},
        },
    },
}

AVG_SPEED_KMH = 18.0  # mountain-road average incl. dwell, assumed


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cumulative_times_s(stop_ids):
    times = [0.0]
    for a, b in zip(stop_ids, stop_ids[1:]):
        lat1, lon1 = STOPS[a][1], STOPS[a][2]
        lat2, lon2 = STOPS[b][1], STOPS[b][2]
        dist = haversine_km(lat1, lon1, lat2, lon2)
        times.append(times[-1] + dist / AVG_SPEED_KMH * 3600)
    return times


def hhmmss(total_seconds):
    total_seconds = int(round(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build():
    os.makedirs(OUT_DIR, exist_ok=True)

    agency_rows = [{
        "agency_id": "ENA",
        "agency_name": "Govern d'Andorra - Bus (Coopalsa / Autocars Nadal)",
        "agency_url": "https://bus.ad/",
        "agency_timezone": "Europe/Andorra",
        "agency_lang": "ca",
        "agency_phone": "+376760100",
    }]

    calendar_rows = [
        {"service_id": "WEEKDAY", "monday": 1, "tuesday": 1, "wednesday": 1,
         "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
         "start_date": "20260101", "end_date": "20261231"},
        {"service_id": "WEEKEND", "monday": 0, "tuesday": 0, "wednesday": 0,
         "thursday": 0, "friday": 0, "saturday": 1, "sunday": 1,
         "start_date": "20260101", "end_date": "20261231"},
        {"service_id": "SATURDAY", "monday": 0, "tuesday": 0, "wednesday": 0,
         "thursday": 0, "friday": 0, "saturday": 1, "sunday": 0,
         "start_date": "20260101", "end_date": "20261231"},
    ]
    day_type_to_service = {
        "weekday": "WEEKDAY", "weekend": "WEEKEND", "saturday": "SATURDAY",
    }

    stops_rows = [
        {"stop_id": sid, "stop_name": name, "stop_lat": lat, "stop_lon": lon}
        for sid, (name, lat, lon) in STOPS.items()
    ]

    routes_rows = []
    trips_rows = []
    stop_times_rows = []
    frequencies_rows = []

    for route_id, route in ROUTES.items():
        routes_rows.append({
            "route_id": route_id,
            "agency_id": "ENA",
            "route_short_name": route_id,
            "route_long_name": route["name"],
            "route_type": 3,  # bus
        })

        for direction_id, stop_ids in enumerate([route["stops"], list(reversed(route["stops"]))]):
            offsets = cumulative_times_s(stop_ids)
            for day_type, windows in route["freq"][direction_id].items():
                service_id = day_type_to_service[day_type]
                trip_id = f"{route_id}_{service_id}_D{direction_id}"
                trips_rows.append({
                    "route_id": route_id,
                    "service_id": service_id,
                    "trip_id": trip_id,
                    "trip_headsign": STOPS[stop_ids[-1]][0],
                    "direction_id": direction_id,
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
                for start, end, headway_min in windows:
                    frequencies_rows.append({
                        "trip_id": trip_id,
                        "start_time": start + ":00",
                        "end_time": end + ":00",
                        "headway_secs": headway_min * 60,
                        "exact_times": 0,
                    })

    def write_csv(name, rows, fieldnames):
        with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_csv("agency.txt", agency_rows,
               ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang", "agency_phone"])
    write_csv("calendar.txt", calendar_rows,
               ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday", "start_date", "end_date"])
    write_csv("stops.txt", stops_rows, ["stop_id", "stop_name", "stop_lat", "stop_lon"])
    write_csv("routes.txt", routes_rows,
               ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"])
    write_csv("trips.txt", trips_rows,
               ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id"])
    write_csv("stop_times.txt", stop_times_rows,
               ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
    write_csv("frequencies.txt", frequencies_rows,
               ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"])

    print(f"Wrote GTFS feed to {OUT_DIR}")
    print(f"  routes={len(routes_rows)} stops={len(stops_rows)} trips={len(trips_rows)} "
          f"stop_times={len(stop_times_rows)} frequencies={len(frequencies_rows)}")


if __name__ == "__main__":
    build()
