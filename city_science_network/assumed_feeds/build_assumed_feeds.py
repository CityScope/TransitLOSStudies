"""Build the ASSUMED supplementary GTFS feeds of the city-science cities (lines that no official feed we could download contains).

Every feed written here is marked so that it can always be told apart from official data:
  * the feed folder is named `assumed_*`; it contains `ASSUMED.md` (sources, method, what is an assumption) and is listed in `assumed_feeds/registry.json`;
  * `feed_info.txt`: feed_publisher_name starts with "ASSUMED";
  * `agency.txt`: agency_name starts with "ASSUMED";
  * `routes.txt`: route_id starts with "assumed_" and route_desc starts with "ASSUMED:".

Feeds
  guadalajara/gtfs/assumed_t21_moovit_osm: Mi Transporte T21/C01 (Central de Autobuses - Tonala - Puente Grande - Zapotlanejo). Stop NAMES and order from the Moovit
      line PDF (65 + 72 stops); stop LOCATIONS = the OpenStreetMap bus stop with the same name chosen by dynamic programming so the stops follow one corridor
      (names that no OSM stop carries are left out); the path between stops follows the OSM road network; headway 20 min, 05:00-21:00 / 06:00-22:00 (El Ciudadano
      Jalisco, launch of the route, 2023).
  concepcion/gtfs/assumed_tome_401_411: Transportes Tome lines 401 (Alto Tome - Concepcion) and 411 (Dichato - Tome - Concepcion), DTPR exclusion-perimeter services
      that the DTPR Gran Concepcion feed does not contain. Terminals from OpenStreetMap, path = shortest road path, stops = OSM bus stops within 35 m of the path.
      Headways: 401 every 5 min at peak / 7 min otherwise (Seremi de Transportes note; Tome bus federation 2020), 411 hourly (Rome2Rio listing).

Usage (from city_science_network/): `python assumed_feeds/build_assumed_feeds.py [guadalajara] [concepcion]`
"""
import csv, hashlib, heapq, json, math, re, subprocess, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parents[2] / "cache" / "assumed_feeds"; CACHE.mkdir(parents=True, exist_ok=True)   # the shared study cache
UA = "transitlos-research/1.0"
HIGHWAYS = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|service"


def overpass(name, query):
    f = CACHE / f"{name}.json"
    if f.is_file():
        return json.loads(f.read_text())
    for _ in range(4):
        out = subprocess.run(["curl", "-s", "-m", "300", "-A", UA, "--data-urlencode", "data=" + query, "https://overpass-api.de/api/interpreter"], capture_output=True, text=True).stdout
        try:
            d = json.loads(out)
            f.write_text(json.dumps(d)); return d
        except Exception:
            import time; time.sleep(20)
    raise RuntimeError(f"overpass failed: {name}")


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(h))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode())


class Roads:
    def __init__(self, name, bbox):
        s, w, n, e = bbox
        d = overpass(f"roads_{name}", f'[out:json][timeout:280];way["highway"~"^({HIGHWAYS})$"]({s},{w},{n},{e});out geom;')
        self.xy, self.adj = {}, {}
        for el in d["elements"]:
            g = el.get("geometry") or []
            ids = el.get("nodes") or []
            for i in range(len(g) - 1):
                a, b = ids[i], ids[i + 1]
                pa, pb = (g[i]["lat"], g[i]["lon"]), (g[i + 1]["lat"], g[i + 1]["lon"])
                self.xy[a], self.xy[b] = pa, pb
                w_ = hav(pa, pb)
                self.adj.setdefault(a, []).append((b, w_)); self.adj.setdefault(b, []).append((a, w_))
        self.ids = list(self.xy)
        self.cells = {}
        for i in self.ids:
            la, lo = self.xy[i]; self.cells.setdefault((int(la * 200), int(lo * 200)), []).append(i)

    def nearest(self, p, rmax=3):
        c0 = (int(p[0] * 200), int(p[1] * 200)); best = None
        for r in range(rmax + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    for i in self.cells.get((c0[0] + dx, c0[1] + dy), []):
                        d = hav(p, self.xy[i])
                        if best is None or d < best[0]:
                            best = (d, i)
            if best and r >= 1:
                break
        return best[1] if best else None

    def path(self, a, b):
        dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                break
            if d > dist.get(u, 1e18):
                continue
            for v, w in self.adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if b not in dist:
            return None
        out = [b]
        while out[-1] != a:
            out.append(prev[out[-1]])
        return [self.xy[i] for i in reversed(out)]


def geocode(q):
    f = CACHE / ("geo_" + norm(q)[:60] + ".json")
    if not f.is_file():
        import time; time.sleep(1.1)
        out = subprocess.run(["curl", "-s", "-m", "60", "-A", UA, "https://nominatim.openstreetmap.org/search", "-G", "--data-urlencode", "q=" + q, "--data-urlencode", "format=json", "--data-urlencode", "limit=1"], capture_output=True, text=True).stdout
        f.write_text(out)
    r = json.loads(f.read_text())
    return (float(r[0]["lat"]), float(r[0]["lon"])) if r else None


MUNI_BOX = {"Tlaquepaque": (20.55, -103.40, 20.68, -103.25), "Tonalá": (20.55, -103.32, 20.72, -103.15), "Zapopan": (20.60, -103.62, 20.90, -103.30),
            "Guadalajara": (20.60, -103.45, 20.76, -103.28), "Tlajomulco": (20.30, -103.60, 20.60, -103.25)}


def accent_regex(name):
    """case-insensitive, accent-insensitive Overpass regex of a place name"""
    plain = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()
    cls = {"a": "[aáàä]", "e": "[eéèë]", "i": "[iíìï]", "o": "[oóòö]", "u": "[uúùü]", "n": "[nñ]"}
    return "".join(cls.get(ch.lower(), re.escape(ch)) for ch in plain)


def locate_in(name, box, near=None):
    """A named place inside a municipality's box: OSM named features (nodes, ways, relations) first, then Nominatim restricted to the box (viewbox, bounded)."""
    s_, w_, n_, e_ = box
    key = "loc_" + norm(name)[:40] + "_%d_%d" % (int(s_ * 100), int(w_ * 100))
    f = CACHE / (key + ".json")
    if not f.is_file():
        pts = []
        try:
            q = '[out:json][timeout:60];nwr["name"~"^%s$",i](%s,%s,%s,%s);out center 12;' % (accent_regex(name), s_, w_, n_, e_)
            out = subprocess.run(["curl", "-s", "-m", "90", "-A", UA, "--data-urlencode", "data=" + q, "https://overpass-api.de/api/interpreter"], capture_output=True, text=True).stdout
            for e in json.loads(out).get("elements", []):
                c = e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")}
                if c.get("lat") is not None:
                    pts.append((c["lat"], c["lon"]))
        except Exception:
            pass
        if not pts:
            import time; time.sleep(1.1)
            out = subprocess.run(["curl", "-s", "-m", "60", "-A", UA, "https://nominatim.openstreetmap.org/search", "-G", "--data-urlencode", "q=" + name, "--data-urlencode", "format=json", "--data-urlencode", "limit=3",
                                  "--data-urlencode", "viewbox=%s,%s,%s,%s" % (w_, n_, e_, s_), "--data-urlencode", "bounded=1"], capture_output=True, text=True).stdout
            try:
                pts = [(float(r["lat"]), float(r["lon"])) for r in json.loads(out)]
            except Exception:
                pts = []
        f.write_text(json.dumps(pts))
    pts = json.loads(f.read_text())
    if not pts:
        return None
    return min(pts, key=lambda c: hav(c, near)) if near else pts[0]


def path_len(pts):
    return sum(hav(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def write_feed(folder, publisher, agency, routes, md):
    """routes: [{id, short, long, desc, shape: [(lat,lon)], stops: [(name, lat, lon)], dist_m: [cum m per stop], windows: [(start, end, headway_min)] , speed_kmh}]"""
    folder.mkdir(parents=True, exist_ok=True)
    def w(name, header, rows):
        with open(folder / name, "w", newline="", encoding="utf-8") as fh:
            c = csv.writer(fh); c.writerow(header); c.writerows(rows)
    w("agency.txt", ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang"], [["assumed", f"ASSUMED - {agency}", "https://github.com", "Asia/Shanghai" if "shanghai" in str(folder) else ("America/Mexico_City" if "guadalajara" in str(folder) else "America/Santiago"), "es"]])
    w("feed_info.txt", ["feed_publisher_name", "feed_publisher_url", "feed_lang", "feed_start_date", "feed_end_date", "feed_version"], [[f"ASSUMED ({publisher})", "https://github.com", "es", "20260101", "20271231", "assumed-1"]])
    w("calendar.txt", ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "start_date", "end_date"], [["ALL", 1, 1, 1, 1, 1, 1, 1, "20260101", "20271231"]])
    stops, rrows, trips, st, fr, shapes = {}, [], [], [], [], []
    for r in routes:
        rrows.append([r["id"], "assumed", r["short"], r["long"], r["desc"], r.get("type", 3)])
        sid = r["id"] + "_shape"
        for i, (la, lo) in enumerate(r["shape"]):
            shapes.append([sid, f"{la:.6f}", f"{lo:.6f}", i + 1])
        for d, (dname, pts) in enumerate(r["directions"]):
            tid = f"{r['id']}_{d}"
            trips.append([r["id"], "ALL", tid, dname, d, sid if d == 0 else ""])
            cum = 0.0
            for k, (name, la, lo, along) in enumerate(pts):
                s_id = f"{r['id']}_{norm(name)[:20]}_{round(la, 5)}_{round(lo, 5)}"
                stops[s_id] = [s_id, f"{name} (ASSUMED location)", f"{la:.6f}", f"{lo:.6f}"]
                t = int(round(along / (r["speed_kmh"] / 3.6)))
                hh = f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
                st.append([tid, hh, hh, s_id, k + 1])
            for (a, b, hw) in r["windows"][d]:
                fr.append([tid, f"{a // 60:02d}:{a % 60:02d}:00", f"{b // 60:02d}:{b % 60:02d}:00", int(hw * 60), 0])
    w("stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon"], list(stops.values()))
    w("routes.txt", ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc", "route_type"], rrows)
    w("trips.txt", ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"], trips)
    w("stop_times.txt", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], st)
    w("frequencies.txt", ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"], fr)
    w("shapes.txt", ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"], shapes)
    (folder / "ASSUMED.md").write_text(md, encoding="utf-8")
    from stagger_phase import stagger_feed   # independent lines must not depart in lockstep at shared stations
    stagger_feed(folder)
    reg_f = Path(__file__).resolve().parent / "registry.json"
    reg = json.loads(reg_f.read_text()) if reg_f.is_file() else {}
    reg[str(folder.relative_to(ROOT))] = {"publisher": publisher, "routes": [r["id"] for r in routes], "marker": "agency_name/feed_publisher_name start with ASSUMED, route_id starts with assumed_, route_desc starts with ASSUMED:"}
    reg_f.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    print(f"wrote {folder}: {len(routes)} routes, {len(stops)} stops")


def along(pts_xy):
    cum = [0.0]
    for i in range(1, len(pts_xy)):
        cum.append(cum[-1] + hav(pts_xy[i - 1], pts_xy[i]))
    return cum


# ------------------------------------------------------------------ Guadalajara T21
def build_guadalajara():
    import os
    # the text of the Moovit T21 line PDF (stop names per direction), saved from moovitapp.com; path in T21_MOOVIT_TXT
    txt = Path(os.environ.get("T21_MOOVIT_TXT", CACHE / "t21_moovit.txt")).read_text()
    a, b = txt.split("Sentido: Zapotlanejo")
    def names(x):
        out = []
        for line in x.splitlines():
            s = line.strip()
            if not s or re.search(r"Horario|lunes|martes|miércoles|jueves|viernes|sábado|domingo|Información|Dirección|Paradas|Duración|Resumen|VER HORARIO|autobús|Usa La App|paradas|Sentido|Los horarios", s):
                continue
            out.append(s)
        return out
    to_ca = names(a)[:]          # Zapotlanejo -> Central de Autobuses (65 stops, in order)
    to_zap = names(b)            # Central de Autobuses -> Zapotlanejo (in order)
    bbox = (20.40, -103.45, 20.75, -103.00)
    d = overpass("stops_gdl", f'[out:json][timeout:200];(node["highway"="bus_stop"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});node["public_transport"="platform"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););out tags center 20000;')
    by = {}
    for e in d["elements"]:
        n = e["tags"].get("name")
        if n:
            by.setdefault(norm(n), []).append((e["lat"], e["lon"]))
    start = geocode("Nueva Central Camionera, Tlaquepaque, Jalisco") or (20.6262, -103.3112)
    end = geocode("Plaza Principal, Zapotlanejo, Jalisco") or (20.6236, -103.0626)
    corridor = (end[0] - start[0], end[1] - start[1]); cl2 = corridor[0] ** 2 + corridor[1] ** 2
    def t_along(p):
        return ((p[0] - start[0]) * corridor[0] + (p[1] - start[1]) * corridor[1]) / cl2
    def dp(seq):
        # candidates per matched name, restricted to the start-end corridor; choose one per name minimising the path length, forward progress preferred
        cand = []
        for n in seq:
            cs = [c for c in by.get(norm(n), []) if -0.05 <= t_along(c) <= 1.05 and abs((c[0] - start[0]) * corridor[1] - (c[1] - start[1]) * corridor[0]) / math.sqrt(cl2) < 0.06]
            if cs:
                cand.append((n, cs))
        best = [{c: (0.0, None) for c in cand[0][1]}] if cand else []
        for i in range(1, len(cand)):
            row = {}
            for c in cand[i][1]:
                row[c] = min(((bp[0] + hav(p, c) + (0 if t_along(c) >= t_along(p) - 0.01 else 4000), p) for p, bp in best[-1].items()), key=lambda x: x[0])
            best.append(row)
        if not best:
            return []
        c = min(best[-1], key=lambda k: best[-1][k][0]); out = [c]
        for i in range(len(best) - 1, 0, -1):
            c = best[i][c][1]; out.append(c)
        out.reverse()
        return [(cand[i][0], out[i][0], out[i][1]) for i in range(len(cand))]
    fwd = dp(to_zap)                                   # outbound stops in order
    print(f"T21: {len(fwd)} of {len(to_zap)} outbound stop names located in OSM, {len(dp(to_ca))} of {len(to_ca)} inbound")
    roads = Roads("gdl_t21", (20.50, -103.40, 20.70, -103.02))
    main = roads.path(roads.nearest(start), roads.nearest(end))
    mcum = along(main)
    def on_path(stops_seq):
        """Matched stops that lie within 150 m of the direct road path, in order of progress along it (the others are dropped)."""
        out, j = [], 0
        for n, la, lo in stops_seq:
            k = min(range(len(main)), key=lambda q: hav(main[q], (la, lo)))
            if hav(main[k], (la, lo)) <= 150:
                out.append((n, main[k][0], main[k][1], mcum[k]))
        return sorted(out, key=lambda r: r[3])
    # stops: OSM bus stops within 60 m of the path (>= 300 m apart); where the path has no stop for > 1 km, interpolated stops every ~700 m
    cand = []
    for e in d["elements"]:
        la, lo = e["lat"], e["lon"]
        k = min(range(len(main)), key=lambda q: hav(main[q], (la, lo))) if abs(la - 20.6) < 0.2 else None
        if k is not None and hav(main[k], (la, lo)) <= 60:
            cand.append((mcum[k], e["tags"].get("name") or "Paradero", la, lo))
    cand.sort()
    picked = [(0.0, "Nueva Central Camionera", *main[0])]
    for al, nm, la, lo in cand:
        if al - picked[-1][0] >= 300:
            picked.append((al, nm, la, lo))
    if mcum[-1] - picked[-1][0] > 300:
        picked.append((mcum[-1], "Plaza Principal Zapotlanejo", *main[-1]))
    filled = []
    for i, p in enumerate(picked):
        if filled and p[0] - filled[-1][0] > 1000:
            base = filled[-1][0]; gap = p[0] - base; n_ = int(gap // 700)
            for q in range(1, n_ + 1):
                al = base + q * gap / (n_ + 1)
                k = min(range(len(main)), key=lambda z: abs(mcum[z] - al))
                filled.append((al, "Paradero (interpolated)", *main[k]))
        filled.append(p)
    filled.sort()
    out_pts = [(n, la, lo, al) for al, n, la, lo in filled]
    in_pts = sorted([(n, la, lo, mcum[-1] - al) for al, n, la, lo in filled], key=lambda r: r[3])
    out_shape = main
    in_shape = list(reversed(main))
    route = {"id": "assumed_T21_C01", "short": "T21/C01", "long": "T21/C01 Central de Autobuses - Tonalá - Puente Grande - Zapotlanejo",
             "desc": "ASSUMED: stops named by Moovit, located at OSM bus stops with the same name; path on OSM roads; 20 min headway from El Ciudadano Jalisco (2023)",
             "shape": out_shape, "speed_kmh": 22.0,
             "directions": [("Zapotlanejo", out_pts), ("Central de Autobuses", in_pts)],
             "windows": [[(6 * 60, 22 * 60, 20)], [(5 * 60, 21 * 60, 20)]]}
    md = (Path(__file__).read_text().split('"""')[1])  # module docstring
    write_feed(ROOT / "guadalajara" / "gtfs" / "assumed_t21_moovit_osm", "Moovit stop names + OpenStreetMap + El Ciudadano Jalisco 2023", "Mi Transporte T21 (assumed)", [route],
               "# ASSUMED feed: Mi Transporte T21/C01\n\nNot official data. " + md + "\nOutbound route length: %.1f km, %d stops located; inbound %d stops.\n" % (path_len(out_shape) / 1000, len(out_pts), len(in_pts)))



# ------------------------------------------------------------------ Guadalajara: OSM routes the official feed lacks
def build_guadalajara_osm_extra():
    import pandas as pd
    d = overpass("rel_gdl_all", '[out:json][timeout:200];relation["route"="bus"](20.30,-103.50,20.90,-103.00);out tags;')
    r = pd.read_csv(ROOT / "guadalajara" / "gtfs" / "mdb-2366_public_tra_mi_transpo" / "routes.txt", dtype=str)
    feed = {norm(v) for v in list(r.route_short_name.fillna("")) + list(r.route_id)}
    miss = {}
    for e in d["elements"]:
        ref = e["tags"].get("ref") or ""
        if ref and norm(ref) not in feed:
            miss.setdefault(ref, []).append(e["id"])
    ids = [i for v in miss.values() for i in v]
    g = overpass("rel_gdl_missing_geom", "[out:json][timeout:250];relation(id:%s);out geom;" % ",".join(map(str, ids)))
    routes = []
    roads = Roads("gdl_all", (20.35, -103.55, 20.85, -102.98))
    for e in g["elements"]:
        tags = e["tags"]; ref = tags.get("ref")
        stops, shape = [], []
        for m in e.get("members", []):
            if m["type"] == "node" and m.get("role", "").startswith(("stop", "platform")) and "lat" in m:
                stops.append((tags.get("name", ref), m["lat"], m["lon"]))
            elif m["type"] == "way" and m.get("geometry") and m.get("role", "") in ("", "route"):
                pts = [(p["lat"], p["lon"]) for p in m["geometry"]]
                if shape and hav(shape[-1], pts[0]) > hav(shape[-1], pts[-1]):
                    pts.reverse()
                shape.extend(pts)
        if len(shape) < 2 and len(stops) >= 2:   # relation holds only the stops (platforms): connect them along the road network
            last = None
            for _, la, lo in stops:
                nid = roads.nearest((la, lo))
                if nid is None:
                    continue
                if last is not None and last != nid:
                    seg = roads.path(last, nid)
                    if seg:
                        shape.extend(seg if not shape else seg[1:])
                elif last is None:
                    shape.append(roads.xy[nid])
                last = nid
        if len(shape) < 2 or len(stops) < 2:
            continue
        cum = along(shape); pts_al, j = [], 0
        for nm, la, lo in stops:
            k = min(range(j, len(shape)), key=lambda q: hav(shape[q], (la, lo))); j = k
            pts_al.append((f"{nm} stop {len(pts_al) + 1}", la, lo, cum[k]))
        routes.append({"id": f"assumed_osm_{norm(ref)}_{e['id']}", "short": ref, "long": tags.get("name", ref),
                       "desc": "ASSUMED: route and stops from OpenStreetMap relation %d; 20 min headway 06:00-21:00 (no timetable found; HEADWAY_BASIS in ASSUMED.md)" % e["id"],
                       "shape": shape, "speed_kmh": 20.0, "directions": [(tags.get("to", "end"), pts_al)], "windows": [[(6 * 60, 21 * 60, 20)]]})
    md = "# ASSUMED feed: Guadalajara routes from OpenStreetMap\n\nNot official data. OSM bus route relations of the Mi Transporte network whose ref is not in the official feed (mdb-2366): %s.\nRoute and stops come from the OSM relation; there is no timetable, so every route has an assumed 20-minute headway from 06:00 to 21:00.\n\nHEADWAY_BASIS: these lines are far less important than the lines of the official feed and must never look better than them, so the value is NOT the feed's own class median (complementary lines 4.8 min, trunk 2.9 min). 20 min is the headway the state itself announces for its newer outer routes: T21 Zapotlanejo-Guadalajara, 12 buses, 20 min (El Ciudadano Jalisco 2023) and a new feeder in north Zapopan, 20 min (Informador 2026-11-04); routes already running at 10-12 min (Tren Ligero feeders) are the better-served ones.\n" % ", ".join(sorted(miss))
    write_feed(ROOT / "guadalajara" / "gtfs" / "assumed_osm_routes", "OpenStreetMap route relations", "Mi Transporte routes from OpenStreetMap (assumed)", routes, md)


# ------------------------------------------------------------------ Concepcion 401 / 411
def build_concepcion():
    bbox = (-36.90, -73.12, -36.50, -72.88)
    d = overpass("stops_conce", f'[out:json][timeout:200];node["highway"="bus_stop"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out tags center 20000;')
    stops = [(e["lat"], e["lon"], e["tags"].get("name") or "Paradero") for e in d["elements"]]
    tome_term = geocode("Terminal de buses, Tomé, Chile") or geocode("Plaza de Armas, Tomé, Chile") or (-36.6180, -72.9560)
    alto_tome = geocode("Los Ciruelos, Tomé, Chile") or (-36.6240, -72.9500)
    dichato = geocode("Dichato, Tomé, Chile") or (-36.5500, -72.9333)
    # Av. Chacabuco / Tucapel intersection in Concepcion (Rome2Rio: departure point of the Tome buses)
    dd = overpass("chacabuco_tucapel", '[out:json][timeout:120];(way["name"~"Chacabuco"](-36.84,-73.07,-36.81,-73.03);)->.a;(way["name"="Tucapel"](-36.84,-73.07,-36.81,-73.03);)->.b;(node(w.a);)->.na;(node(w.b);)->.nb;node.na.nb;out;')
    conce = (dd["elements"][0]["lat"], dd["elements"][0]["lon"]) if dd["elements"] else (-36.8283, -73.0453)
    roads = Roads("conce_tome", (-36.90, -73.10, -36.52, -72.90))
    def line(points, hail):
        shape, last = [], None
        for p in points:
            nid = roads.nearest(p)
            if last is not None:
                seg = roads.path(last, nid)
                if seg is None:
                    raise RuntimeError("no road path between terminals")
                shape.extend(seg if not shape else seg[1:])
            else:
                shape.append(roads.xy[nid])
            last = nid
        return shape
    def stops_on(shape, terminals):
        cum = along(shape); res = [("Terminal", shape[0][0], shape[0][1], 0.0)]
        last_k = 0
        for (la, lo, nm) in stops:
            best = min(range(last_k, len(shape)), key=lambda k: hav((la, lo), shape[k]))
            if hav((la, lo), shape[best]) <= 35 and cum[best] - res[-1][3] >= 150:
                res.append((nm, la, lo, cum[best])); last_k = best
        res.append(("Terminal", shape[-1][0], shape[-1][1], cum[-1]))
        res.sort(key=lambda r: r[3])
        filled = []   # gaps of more than 1 km between OSM stops: interpolated stops every ~700 m (marked in the stop name)
        for r_ in res:
            if filled and r_[3] - filled[-1][3] > 1000:
                base = filled[-1][3]; gap = r_[3] - base; n_ = int(gap // 700)
                for q in range(1, n_ + 1):
                    al = base + q * gap / (n_ + 1); k = min(range(len(shape)), key=lambda z: abs(cum[z] - al))
                    filled.append(("Paradero (interpolated)", shape[k][0], shape[k][1], al))
            filled.append(r_)
        return filled
    # 401: Alto Tome -> Tome terminal -> Concepcion; 411: Dichato -> Tome -> Concepcion (the 411 shares the Tome-Concepcion corridor)
    s401 = line([alto_tome, tome_term, conce], 0)
    s411 = line([dichato, tome_term, conce], 0)
    def rev(pts, shape):
        L = along(shape)[-1]
        return sorted([(n, la, lo, L - a) for n, la, lo, a in pts], key=lambda r: r[3])
    p401, p411 = stops_on(s401, None), stops_on(s411, None)
    r401 = {"id": "assumed_401", "short": "401", "long": "401 Alto Tomé - Concepción (Transportes Tomé)",
            "desc": "ASSUMED: terminals from OpenStreetMap, road path, OSM bus stops within 35 m; 5 min peak / 7 min off-peak (Seremi de Transportes, Tomé bus federation)",
            "shape": s401, "speed_kmh": 28.0, "directions": [("Concepción", p401), ("Alto Tomé", rev(p401, s401))],
            "windows": [[(5 * 60 + 30, 8 * 60 + 30, 5), (8 * 60 + 30, 17 * 60, 7), (17 * 60, 20 * 60 + 30, 5), (20 * 60 + 30, 22 * 60, 10)],
                        [(5 * 60 + 30, 8 * 60 + 30, 5), (8 * 60 + 30, 17 * 60, 7), (17 * 60, 20 * 60 + 30, 5), (20 * 60 + 30, 22 * 60, 10)]]}
    r411 = {"id": "assumed_411", "short": "411", "long": "411 Dichato - Tomé - Concepción (Transportes Tomé)",
            "desc": "ASSUMED: terminals from OpenStreetMap, road path, OSM bus stops within 35 m; hourly (Rome2Rio listing)",
            "shape": s411, "speed_kmh": 28.0, "directions": [("Concepción", p411), ("Dichato", rev(p411, s411))],
            "windows": [[(6 * 60, 21 * 60, 60)], [(6 * 60, 21 * 60, 60)]]}
    md = "# ASSUMED feed: Transportes Tomé 401 and 411\n\nNot official data.\n" + Path(__file__).read_text().split('"""')[1]
    write_feed(ROOT / "concepcion" / "gtfs" / "assumed_tome_401_411", "OpenStreetMap + Seremi Transportes + Tomé bus federation + Rome2Rio", "Transportes Tome 401/411 (assumed)", [r401, r411], md)
    # Nueva Los Tilos (49 buses): Tomé - Concepción every 30 min and Tomé - Coliumo every 45 min (spokesman of the line, El Conquistador 2023-07-25)
    coliumo = geocode("Coliumo, Tomé, Chile") or (-36.5430, -72.9540)
    snl_c = line([tome_term, conce], 0)
    snl_k = line([coliumo, tome_term], 0)
    pc, pk = stops_on(snl_c, None), stops_on(snl_k, None)
    day = lambda h: [[(5 * 60 + 30, 22 * 60, h)], [(5 * 60 + 30, 22 * 60, h)]]
    rc = {"id": "assumed_nlt_centro", "short": "NLT", "long": "Nueva Los Tilos Tomé - Concepción",
          "desc": "ASSUMED: terminals from OpenStreetMap, road path, OSM bus stops within 35 m; every 30 min (spokesman of the line, El Conquistador, 25 Jul 2023); hours assumed like the 401",
          "shape": snl_c, "speed_kmh": 28.0, "directions": [("Concepción", pc), ("Tomé", rev(pc, snl_c))], "windows": day(30)}
    rk = {"id": "assumed_nlt_coliumo", "short": "NLT-C", "long": "Nueva Los Tilos Coliumo - Tomé",
          "desc": "ASSUMED: terminals from OpenStreetMap, road path, OSM bus stops within 35 m; every 45 min (spokesman of the line, El Conquistador, 25 Jul 2023); hours assumed like the 401",
          "shape": snl_k, "speed_kmh": 28.0, "directions": [("Tomé", pk), ("Coliumo", rev(pk, snl_k))], "windows": day(45)}
    md2 = ("# ASSUMED feed: Nueva Los Tilos (Tomé)\n\nNot official data.\nThe operator (49 buses in the 2024 fleet list) runs Tomé - Concepción every 30 min and Tomé - Coliumo every 45 min "
           "(spokesman, El Conquistador 2023-07-25: https://www.elconquistadorconcepcion.cl/2023/07/25/choferes-de-la-linea-nueva-los-tilos-de-tome-depuso-movilizacion-tras-inicio-de-mesa-de-trabajo/). "
           "No published stop list: terminals and paths from OpenStreetMap as for the 401/411; operating hours assumed.\n\n"
           "Not covered: Los Tilos (10 buses) and Ruta Costa (4): no route, terminal or frequency found in any source searched.\n")
    write_feed(ROOT / "concepcion" / "gtfs" / "assumed_nueva_los_tilos", "OpenStreetMap + El Conquistador (Concepción)", "Nueva Los Tilos (assumed)", [rc, rk], md2)


# ------------------------------------------------------------------ Concepcion: OSM inter-town / rural routes the DTPR feed lacks
#: fleet per operator, "Flota P.E. Gran Concepcion, Lota - Coronel y Tome 2024" v3 (community spreadsheet built from the MTT/DTPR fleet register, updated 2024-08-10)
FLEET_2024 = {"302": 88, "303": 52, "304": 21, "300": 39, "local": 44 + 39 + 43 + 28, "44": 38}   # local = Sotral 44 + Nueva Takora 39 + Lota 160 43 + Playa Colcura 28


def fleet_headway(fleet, round_trip_km, speed_kmh, layover=0.10):
    """Headway (min) = round-trip time / buses in service; buses in service = fleet - MAINTENANCE_SHARE - SPARE_SHARE (8% workshop, 9% reserve); layover 10% of the running time at terminals."""
    in_service = fleet * (1 - 0.08 - 0.09)
    return max(2, min(60, round(round_trip_km / speed_kmh * 60 * (1 + layover) / in_service)))


def apply_fleet_headways(routes):
    """Fleet-based headways for the Lota-Coronel services and line 44, and the clones of the 300 corridor for lines 302, 303 and 304.
    Wikipedia (Transporte publico en el Gran Concepcion) lists 300, 302, 303 and 304 as the four regulated services on the same corridor (Parque Lota/Lota - Coronel - San Pedro de la Paz - Collao, Concepcion),
    so 302/303/304 reuse the OSM geometry of the 300; each gets the headway its own fleet implies. Local Coronel-Lota services (Sotral, Nueva Takora, Nueva Ruta 160, Playa Colcura) share the
    OSM 'Coronel-Lota (Vuelta)' relations: their combined fleet is spread over those branches in proportion to branch length (all branches get the same headway)."""
    def length_km(r):
        return along(r["shape"])[-1] / 1000
    local_r = []
    out, c300 = [], [r for r in routes if str(r["short"]).startswith("300")]
    local = [r for r in routes if "Coronel-Lota" in str(r["short"])]
    l44 = [r for r in routes if str(r["short"]).startswith("44")]
    rest = [r for r in routes if r not in c300 + local + l44]
    if c300:
        rt = sum(length_km(r) for r in c300)   # ida + regreso = one round trip
        for r in c300:   # the 300 keeps the Seremi's line-specific 3-5 min (upper end); the fleet value is printed as a check
            print(f"    300: fleet check {FLEET_2024['300']} buses -> {fleet_headway(FLEET_2024['300'], rt, 25.0)} min (Seremi guide: 3-5 min; used 5)")
        for ln in ("302", "303", "304"):
            hw = fleet_headway(FLEET_2024[ln], rt, 25.0)
            print(f"    {ln}: {FLEET_2024[ln]} buses, corridor round trip {rt:.0f} km -> every {hw} min")
            for r in c300:
                c = dict(r, id=r["id"].replace("assumed_osm_300", f"assumed_osm_{ln}"), short=r["short"].replace("300", ln), long=r["long"].replace("300", ln),
                         desc=f"ASSUMED: corridor geometry of OSM relation (same route as the 300, Wikipedia); headway {hw} min from the {FLEET_2024[ln]}-bus fleet of line {ln} (fleet register 2024: 8% workshop, 9% reserve), 05:00-22:00",
                         windows=[[(5 * 60, 22 * 60, hw)]])
                out.append(c)
    if local and c300:
        # The OSM 'Coronel-Lota (Vuelta)' relations are 0.5-1.9 km fragments inside Coronel, no usable route. The local Coronel-Lota services (Sotral, Nueva Takora, Nueva Ruta 160,
        # Playa Colcura) run on the Lota-Coronel stretch of the same road, so that stretch of the 300 geometry (Lota -> Coronel centre) is used, in both directions.
        CORONEL = (-37.0285, -73.1580)
        for r in c300:
            sh = r["shape"]
            if "ida" in r["short"]:      # Lota -> Concepcion: the stretch from Lota up to the point nearest Coronel centre
                k = min(range(len(sh)), key=lambda q: hav(sh[q], CORONEL)); seg_sh = sh[:k + 1]
            else:                        # Concepcion -> Lota: from Coronel centre to Lota
                k = min(range(len(sh)), key=lambda q: hav(sh[q], CORONEL)); seg_sh = sh[k:]
            if len(seg_sh) < 3:
                continue
            cum0 = along(sh); lo, hi = (0.0, cum0[k]) if "ida" in r["short"] else (cum0[k], cum0[-1])
            dirs = [(h_, [(n, la, lo_, a_ - lo) for (n, la, lo_, a_) in st if lo <= a_ <= hi]) for h_, st in r["directions"]]
            local_r.append(dict(r, id=r["id"].replace("assumed_osm_300", "assumed_osm_local"), short="Coronel-Lota " + ("Lota-Coronel" if "ida" in r["short"] else "Coronel-Lota"),
                                long="Coronel - Lota local service (Sotral, Nueva Takora, Nueva Ruta 160, Playa Colcura)", shape=seg_sh, directions=dirs))
        L = sum(length_km(r) for r in local_r)   # both directions = one round trip
        fh = fleet_headway(FLEET_2024["local"], L, 20.0)
        hw = max(3, fh)
        print(f"    Coronel-Lota local: {FLEET_2024['local']} buses, stretch round trip {L:.0f} km -> fleet value {fh} min, used {hw} min (floor 3 min: the four operators also run Coronel/Lota neighbourhood branches and part of the fleet runs to Concepcion)")
        for r in local_r:
            r["windows"] = [[(6 * 60, 22 * 60, hw)]]
            r["speed_kmh"] = 20.0
            r["desc"] = f"ASSUMED: Lota-Coronel stretch of the corridor geometry (OSM relation of the 300); headway {hw} min (fleet of the four local operators 154 buses -> {fh} min, floored at 3 min), 06:00-22:00"
        local = local_r
    elif local:
        local = []
    if l44:
        ab = [r for r in l44 if r["short"][2:3] in "AB" and len(r["short"]) <= 3]
        rt = sum(length_km(r) for r in ab) + sum(2 * length_km(r) for r in l44 if r not in ab)
        hw = fleet_headway(FLEET_2024["44"], rt, 20.0)
        print(f"    44 (Flota Centauro): {FLEET_2024['44']} buses, {len(l44)} variants ({rt:.0f} km of round trips) -> every {hw} min")
        for r in l44:
            r["windows"] = [[(5 * 60 + 30, 22 * 60, hw)]]
            r["desc"] = f"ASSUMED: OSM relation of line 44 (Flota Centauro); headway {hw} min from its 38-bus fleet (fleet register 2024)"
            r["speed_kmh"] = 20.0
    return rest + c300 + out + local + l44


def build_concepcion_osm_extra():
    import pandas as pd
    bbox = (-37.25, -73.30, -36.55, -72.60)
    d = overpass("rel_conce_all", '[out:json][timeout:120];(relation["route"~"bus|minibus"](%s,%s,%s,%s);relation["route"="share_taxi"](%s,%s,%s,%s););out tags 400;' % (bbox + bbox))
    r = pd.read_csv(ROOT / "concepcion" / "gtfs" / "gran_concepcion" / "routes.txt", dtype=str)
    feed = {norm(v) for v in r.route_short_name.fillna("")}
    feed_nums = {re.match(r"\d+", v).group() for v in r.route_short_name.fillna("") if re.match(r"\d+", v)}
    dtpr_code = re.compile(r"^\d{2}[A-Za-z]$|^B0\d")
    ids, names = [], []
    for e in d["elements"]:
        ref = e["tags"].get("ref") or ""
        if (ref and norm(ref) in feed) or (dtpr_code.match(ref.split(" ")[0] if ref else "x") and (ref[:2] in feed_nums or ref.startswith("B0"))):
            continue   # in the feed, or a letter variant of a DTPR line the feed already has
        ids.append(e["id"]); names.append(e["tags"].get("name") or ref)
    g = overpass("rel_conce_missing_geom_" + __import__("hashlib").md5(",".join(map(str, sorted(ids))).encode()).hexdigest()[:8], "[out:json][timeout:250];relation(id:%s);out geom;" % ",".join(map(str, ids)))
    roads = Roads("conce_all", (-37.20, -73.25, -36.60, -72.70))
    routes = []
    for e in g["elements"]:
        tags = e["tags"]; ref = tags.get("ref") or tags.get("name")
        stops, shape = [], []
        for m in e.get("members", []):
            if m["type"] == "node" and m.get("role", "").startswith(("stop", "platform")) and "lat" in m:
                stops.append((tags.get("name", ref), m["lat"], m["lon"]))
            elif m["type"] == "way" and m.get("geometry") and m.get("role", "") in ("", "route"):
                pts = [(p["lat"], p["lon"]) for p in m["geometry"]]
                if shape and hav(shape[-1], pts[0]) > hav(shape[-1], pts[-1]):
                    pts.reverse()
                shape.extend(pts)
        if len(shape) < 2:
            continue
        if len(stops) < 2:   # no stop members: a stop at both ends and one every ~600 m along the way
            cum = along(shape); stops = [(tags.get("name", ref), *shape[0])]
            nxt = 600.0
            for q in range(1, len(shape)):
                if cum[q] >= nxt:
                    stops.append((tags.get("name", ref), *shape[q])); nxt += 600.0
            stops.append((tags.get("name", ref), *shape[-1]))
        cum = along(shape); pts_al, j = [], 0
        for nm, la, lo in stops:
            k = min(range(j, len(shape)), key=lambda q: hav(shape[q], (la, lo))); j = k
            pts_al.append((f"{nm} stop {len(pts_al) + 1}", la, lo, cum[k]))
        rural = str(tags.get("ref", "")).startswith("CTR")
        if rural:      # subsidised rural services: Caleta Chome 2 days a week (MTT); Hualqui-Talcamavida has no published timetable -> 3 trips a day
            hw, win = (720, (7 * 60, 19 * 60)) if "0222" in str(tags.get("ref")) else (120, (7 * 60, 19 * 60))
        elif str(tags.get("ref", "")).startswith("300"):   # Lota-Concepcion: Seremi guide, 3-5 min after the first departure (line-specific); the upper end is used
            hw, win = 5, (5 * 60, 22 * 60)
        else:          # Coronel-Lota loops (Sotral, new 2023 urban line): cautious 20 min
            hw, win = 20, (6 * 60, 22 * 60)
        routes.append({"id": f"assumed_osm_{norm(ref)}_{e['id']}", "short": ref, "long": tags.get("name", ref),
                       "desc": "ASSUMED: route and stops from OpenStreetMap relation %d; headway %d min %02d:00-%02d:00 (HEADWAY_BASIS in ASSUMED.md)" % (e["id"], hw, win[0] // 60, win[1] // 60),
                       "shape": shape, "speed_kmh": 25.0, "directions": [(tags.get("to", "end"), pts_al)],
                       "windows": [[(win[0], win[1], hw)]]})
    routes = apply_fleet_headways(routes)
    md = ("# ASSUMED feed: Gran Concepción routes from OpenStreetMap + fleet register\n\nNot official data. Services that the DTPR Gran Concepción feed does not contain: it has 6 stops in Coronel and none in Lota, Tomé or Santa Juana, and no line 44.\n"
          "Routes written: %s.\n\n"
          "GEOMETRY: OSM route relations (ways + stop members; one stop every 600 m when the relation has no stops). Relations that are letter variants of a DTPR line already in the feed are skipped; line 44 (Flota Centauro, 44A/44B/44E) is not in the feed, so it is kept.\n"
          "The OSM 'Coronel-Lota (Vuelta)' relations are 0.5-1.9 km fragments and are not used.\n\n"
          "HEADWAY_BASIS (never the feed's medians):\n"
          "- 300 Lota-Concepción: Seremi de Transportes guide, 3-5 min after the first departure (Lota 05:07, Concepción 05:20; 2023): 5 min, 05:00-22:00. The fleet value (39 buses) would give 8 min.\n"
          "- 302, 303, 304: same corridor as the 300 (Wikipedia, 'Transporte público en el Gran Concepción': Parque Lota/Lota - Coronel - San Pedro de la Paz - Collao); geometry cloned from the 300. Headway = round-trip time / buses in service, from the fleet register "
          "(Flota P.E. Gran Concepción, Lota - Coronel y Tomé 2024 v3, 2024-08-10): 302 Sotral 88 buses -> 3 min, 303 Playa Blanca 52 -> 6 min, 304 Expresos del Carbón 21 -> 15 min. 8%% of the fleet is in the workshop and 9%% is spare; layover 10%% of running time; 25 km/h.\n"
          "- Local Coronel-Lota (Sotral 44, Nueva Takora 39, Nueva Ruta 160 43, Playa Colcura 28 = 154 buses): the Lota-Coronel stretch of the 300 geometry (22 km round trip) at 20 km/h; the fleet alone gives 2 min, floored at 3 min "
          "because these operators also run neighbourhood branches and part of the fleet goes on to Concepción. 06:00-22:00 (Sotral night run 22:00 from Lota Bajo, 2026; Playa Colcura 07:00-20:00, Subtrans 2023).\n"
          "- 44 Flota Centauro: 38 buses over the three OSM variants (88 km of round trips) at 20 km/h -> 9 min, 05:30-22:00.\n"
          "- Rural CTR0222 Talcahuano-Caleta Chome: MTT, the subsidised bus reaches the cove twice a week -> one trip every 12 h. CTR0201 Hualqui-Talcamávida: no timetable found, 3 trips a day (assumption).\n"
          "Not covered (no route or fleet-to-line data found): Los Tilos 10 buses, Nueva Los Tilos 49, Ruta Costa 4.\n") % ", ".join(sorted(set(names)))
    write_feed(ROOT / "concepcion" / "gtfs" / "assumed_osm_routes", "OpenStreetMap route relations", "Gran Concepcion inter-town and rural routes from OpenStreetMap (assumed)", routes, md)


# ------------------------------------------------------------------ Concepcion: Santa Juana - Concepcion (regulated system, fleet-based headway)
MAINTENANCE_SHARE, SPARE_SHARE = 0.08, 0.09     # buses in the workshop / kept as reserve (same shares as the Shanghai fleet check)


def build_concepcion_santa_juana():
    """The MTT/DTPR 2025 sheet lists a regulated, GPS-monitored system of 23 buses (6 high-standard) linking Santa Juana with Concepcion; no timetable exists online.
    Headway = round-trip time / buses in service, with 8% of the fleet in maintenance and 9% held as spare, and 20% of the in-service buses left for the system's local branches."""
    bbox = (-37.25, -73.12, -36.78, -72.85)
    d = overpass("stops_santajuana", f'[out:json][timeout:200];node["highway"="bus_stop"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out tags center 20000;')
    stops = [(e["lat"], e["lon"], e["tags"].get("name") or "Paradero") for e in d["elements"]]
    sj = geocode("Plaza de Armas, Santa Juana, Chile") or (-37.1730, -72.9400)
    dd = overpass("chacabuco_tucapel", '[out:json][timeout:120];(way["name"~"Chacabuco"](-36.84,-73.07,-36.81,-73.03);)->.a;(way["name"="Tucapel"](-36.84,-73.07,-36.81,-73.03);)->.b;(node(w.a);)->.na;(node(w.b);)->.nb;node.na.nb;out;')
    conce = (dd["elements"][0]["lat"], dd["elements"][0]["lon"]) if dd["elements"] else (-36.8283, -73.0453)
    roads = Roads("conce_sj", bbox)
    shape, last = [], None
    for p_ in (sj, conce):
        nid = roads.nearest(p_)
        if last is not None:
            seg = roads.path(last, nid)
            if seg is None:
                raise RuntimeError("no road path Santa Juana - Concepcion")
            shape.extend(seg)
        else:
            shape.append(roads.xy[nid])
        last = nid
    cum = along(shape); L = cum[-1]
    pts, last_k = [("Terminal Santa Juana", shape[0][0], shape[0][1], 0.0)], 0
    for (la, lo, nm) in stops:
        best = min(range(last_k, len(shape)), key=lambda k: hav((la, lo), shape[k]))
        if hav((la, lo), shape[best]) <= 35 and cum[best] - pts[-1][3] >= 150:
            pts.append((nm, la, lo, cum[best])); last_k = best
    pts.append(("Concepción centro", shape[-1][0], shape[-1][1], L))
    filled = []
    for r_ in sorted(pts, key=lambda r: r[3]):
        if filled and r_[3] - filled[-1][3] > 1000:
            base = filled[-1][3]; gap = r_[3] - base; n_ = int(gap // 700)
            for q in range(1, n_ + 1):
                al = base + q * gap / (n_ + 1); k = min(range(len(shape)), key=lambda z: abs(cum[z] - al))
                filled.append(("Paradero (interpolated)", shape[k][0], shape[k][1], al))
        filled.append(r_)
    speed = 35.0     # rural highway + urban approach
    run_min = L / 1000 / speed * 60
    fleet = 23
    in_service = fleet * (1 - MAINTENANCE_SHARE - SPARE_SHARE)
    trunk = in_service * 0.8
    hw = max(5, round((2 * run_min + 20) / trunk))   # 20 min layover per round trip
    rev = sorted([(n, la, lo, L - a) for n, la, lo, a in filled], key=lambda r: r[3])
    r = {"id": "assumed_santa_juana", "short": "SJ", "long": "Santa Juana - Concepción (regulated system, 23 buses)",
         "desc": "ASSUMED: terminals from OpenStreetMap, road path, OSM bus stops within 35 m; headway %d min from the 23-bus fleet (DTPR 2025)" % hw,
         "shape": shape, "speed_kmh": speed, "directions": [("Concepción", filled), ("Santa Juana", rev)], "windows": [[(5 * 60 + 30, 21 * 60 + 30, hw)], [(5 * 60 + 30, 21 * 60 + 30, hw)]]}
    md = ("# ASSUMED feed: Santa Juana - Concepción\n\nNot official data. The Subsecretaría de Transportes / DTPR Biobío 2025 sheet lists a regulated, GPS-monitored system of 23 buses (6 high-standard) "
          "that connects Santa Juana (about 14,000 people) with Concepción; no timetable is published online.\n\nHEADWAY_BASIS: route length %.1f km at %.0f km/h = %.0f min one way; round trip + 20 min layover = %.0f min. "
          "Fleet 23 buses minus %.0f%% in maintenance and %.0f%% spare = %.1f in service; 80%% of them (%.1f) on the Concepción trunk, the rest on local branches -> one bus every %d min, 05:30-21:30. "
          "Route and stops: OpenStreetMap (no OSM relation exists for this system, so the path is the road route between the terminals).\n") % (
          L / 1000, speed, run_min, 2 * run_min + 20, MAINTENANCE_SHARE * 100, SPARE_SHARE * 100, in_service, trunk, hw)
    write_feed(ROOT / "concepcion" / "gtfs" / "assumed_santa_juana", "OpenStreetMap + DTPR Biobío 2025 fleet", "Santa Juana - Concepción (assumed)", [r], md)
    print(f"Santa Juana line: {L / 1000:.1f} km, {run_min:.0f} min one way, headway {hw} min, {len(filled)} stops")


# ------------------------------------------------------------------ Guadalajara: lines of the Mi Transporte network that the official feed lacks (treep.app timetables)
#: weekday timetables read from the treep.app trip pages of the Mi Transporte (agency 35) lines that are in neither the official GTFS (mdb-2366) nor OSM:
#: first/last departure and the interval of the printed departure list. The path is built from the named anchor places (geocoded) on the OSM road network.
TREEP_LINES = [
    {"id": "C52", "long": "C52-A Dos Templos - San Martín de las Flores - Tlaquepaque - Guadalajara (647-A)", "run_min": 34, "area": "Tlaquepaque",
     "names": ["Porfirio Díaz", "Tlaquepaque Centro", "Pila Seca", "Plaza Fórum", "Río Nilo", "Calzada Olímpica", "Escuela Vocacional UdeG", "Calzada Del Ejercito", "Heróico Colegio Militar", "Matamoros", "Burgos", "Calzada Independencia", "Terminal De La 647 Y 647-A"],
     "windows": [(5 * 60, 23 * 60, 5), (5 * 60, 23 * 60, 5)]},
    {"id": "C87", "long": "C87 Dos Templos - San Martín de las Flores - Tonalá - Tlaquepaque - Guadalajara (647)", "run_min": 36, "area": "Tlaquepaque",
     "names": ["Porfirio Díaz", "Calzada Delicias", "Tonalá", "Tlaquepaque Centro", "Pila Seca", "Plaza Fórum", "Río Nilo", "Calzada Olímpica", "Heróico Colegio Militar", "Calzada Independencia", "Terminal De La 647 Y 647-A"],
     "windows": [(4 * 60 + 40, 21 * 60 + 30, 10), (5 * 60, 22 * 60 + 30, 10)]},
    {"id": "C84", "long": "C84 Santa Lucía - Tesistán - Zapopan Centro - Plaza Patria - San Jorge (631)", "run_min": 103, "area": "Zapopan",
     "names": ["Santa Lucía", "Salto Del Agua", "Hacienda Las Palomas", "Cucba", "El Triángulo", "Gran Vía Olivos", "Tecnológico de Monterrey", "Recinto de la Paz", "Valle Real", "Zapopan Centro", "Jacarandas", "Obelisco", "San Jorge"],
     "windows": [(5 * 60, 21 * 60, 10), (6 * 60, 22 * 60, 10)]},
    {"id": "C121A", "long": "C121A Guadalajara (Av. Guadalajara) - Zapopan Centro - Patria - Plaza México - Ignacio L. Vallarta (635-A)", "run_min": 92, "area": "Zapopan",
     "names": ["Arco de la Estrella", "Arcos de Zapopan", "Periférico Belenes", "Zapopan Centro", "Jacarandas", "Obelisco", "San Jorge", "Plaza Patria", "Estadio Tres De Marzo", "Plaza México", "Ignacio L. Vallarta"],
     "windows": [(5 * 60, 21 * 60, 20), (6 * 60, 22 * 60, 20)]},
    {"id": "C81", "long": "C81 Mesa de los Ocotes - Zapopan Centro - Centro Cultural Universitario (641)", "run_min": 106, "area": "Zapopan",
     "names": ["Crucero Mesa De Los Ocotes", "Periférico Norte", "Tec Milenio", "Belenes", "Cucea", "Plaza San Isidro", "Costco San Isidro", "Centro Cultural Universitario", "Auditorio Telmex", "Mercado del Mar", "Obelisco", "Jacarandas", "Zapopan Centro", "Lago Ginebra"],
     "windows": [(5 * 60, 21 * 60 + 20, 5), (5 * 60, 21 * 60, 10)]},
    {"id": "T22-V02", "long": "T22-V02 Centro Metropolitano - Santa Cruz de las Flores (Adolfo López Mateos Sur)", "run_min": 111, "area": "Tlajomulco",
     "names": ["Glorieta Minerva", "Punto Sur", "Vista Sur", "Circuito Metropolitano Sur", "Santa Cruz de las Flores"],
     "windows": [(5 * 60, 23 * 60, 30), (5 * 60, 23 * 60, 30)]},
]
#: queries that the geocoder resolves for the named stops (the stop names themselves are often too ambiguous)
GEOQ = {"Tlaquepaque Centro": "Tlaquepaque Centro, Jalisco", "Plaza Fórum": "Plaza Forum Tlaquepaque", "Calzada Olímpica": "Calzada Olímpica, Guadalajara",
        "Heróico Colegio Militar": "Heroico Colegio Militar, Guadalajara", "Zapopan Centro": "Zapopan Centro", "Obelisco": "Obelisco, Zapopan", "Plaza Patria": "Plaza Patria, Zapopan",
        "Estadio Tres De Marzo": "Estadio Tres de Marzo, Zapopan", "Plaza México": "Plaza México, Guadalajara", "Arcos de Zapopan": "Arcos de Zapopan",
        "Punto Sur": "Punto Sur, Tlajomulco", "Vista Sur": "Vista Sur, Tlajomulco", "Hacienda Las Palomas": "Hacienda Las Palomas, Zapopan", "Cucba": "CUCBA, Las Agujas, Zapopan",
        "Glorieta Minerva": "Glorieta Minerva, Guadalajara", "Periférico Belenes": "Periférico Belenes, Zapopan", "Auditorio Telmex": "Auditorio Telmex, Zapopan",
        "Centro Cultural Universitario": "Centro Cultural Universitario, Guadalajara", "Belenes": "Belenes, Zapopan", "Mercado del Mar": "Mercado del Mar, Zapopan",
        "Porfirio Díaz": "Calle Porfirio Díaz, San Martín de las Flores, Tlaquepaque", "Calzada Delicias": "Calzada Delicias, Tonalá", "Santa Cruz de las Flores": "Santa Cruz de las Flores, Tlajomulco de Zúñiga"}
#: municipality of a named stop when it differs from its line's main one (the stop name is searched inside that municipality only)
MUNI = {"Porfirio Díaz": "Tlaquepaque", "Calzada Delicias": "Tlaquepaque", "Tonalá": "Tonalá", "Tlaquepaque Centro": "Tlaquepaque", "Pila Seca": "Tlaquepaque", "Plaza Fórum": "Tlaquepaque",
        "Río Nilo": "Guadalajara", "Calzada Olímpica": "Guadalajara", "Escuela Vocacional UdeG": "Guadalajara", "Calzada Del Ejercito": "Guadalajara", "Heróico Colegio Militar": "Guadalajara",
        "Matamoros": "Guadalajara", "Burgos": "Guadalajara", "Calzada Independencia": "Guadalajara", "Terminal De La 647 Y 647-A": "Guadalajara", "Glorieta Minerva": "Guadalajara",
        "Punto Sur": "Tlajomulco", "Vista Sur": "Tlajomulco", "Circuito Metropolitano Sur": "Tlajomulco", "Santa Cruz de las Flores": "Tlajomulco", "Centro Cultural Universitario": "Guadalajara",
        "Plaza México": "Guadalajara", "Ignacio L. Vallarta": "Guadalajara", "Auditorio Telmex": "Zapopan"}
#: C72 (Atemajac del Valle - Cañón de las Flores): treep.app's stop ids carry coordinates, decoded here (every 8th stop of the 84)
C72_ANCHORS = [(20.721168, -103.363688), (20.722488, -103.381872), (20.723037, -103.398397), (20.731575, -103.406566), (20.730532, -103.418921), (20.729972, -103.432066),
               (20.730430, -103.444615), (20.732876, -103.460228), (20.735794, -103.474304), (20.739560, -103.491180), (20.745888, -103.499477), (20.764881, -103.521292),
               (20.755185, -103.535011), (20.739128, -103.538864), (20.728073, -103.544135), (20.728593, -103.554244), (20.719036, -103.567542)]


def build_guadalajara_treep():
    bbox = (20.40, -103.58, 20.80, -103.00)
    d = overpass("stops_gdl_treep", f'[out:json][timeout:200];node["highway"="bus_stop"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out tags center 40000;')
    osm_stops = [(e["lat"], e["lon"], e["tags"].get("name") or "Paradero") for e in d["elements"]]
    by = {}
    for la, lo, nm in osm_stops:
        by.setdefault(norm(nm), []).append((la, lo))
    roads = Roads("gdl_treep", bbox)

    def resolve(names, area):
        """Anchors in route order: a stop of that name in OSM (the candidate nearest to the previous anchor, hops <= 9 km), else a geocoded place of that name."""
        out = []
        for nm in names:
            cands = by.get(norm(nm), [])
            if out:
                cands = [c for c in cands if hav(c, out[-1]) <= 9000]
                best = min(cands, key=lambda c: hav(c, out[-1])) if cands else None
            else:
                best = cands[0] if len(cands) == 1 else None
            if best is None:
                plain = unicodedata.normalize("NFD", nm).encode("ascii", "ignore").decode()
                for q in (GEOQ.get(nm, ""), f"{nm}, {area}, Jalisco", f"{nm}, Jalisco", f"{plain} {area}", f"{plain}, Guadalajara", f"{nm}, {area}"):
                    g = geocode(q) if q else None
                    if g and 20.40 <= g[0] <= 20.80 and -103.58 <= g[1] <= -103.00 and (not out or hav(g, out[-1]) <= 12000):
                        best = g
                        break
            if best is None:
                box = MUNI_BOX.get(MUNI.get(nm, area))
                g = locate_in(nm, box, near=out[-1] if out else None) if box else None
                if g and (not out or hav(g, out[-1]) <= 12000):
                    best = g
            if best is not None and (not out or hav(best, out[-1]) > 150):
                out.append(best)
            elif best is None:
                print(f"    anchor not located: {nm}")
        return out

    routes = []
    tstops = json.load(open(Path(__file__).with_name("treep_stops.json")))
    TREEP_LINES.extend(dict(x, windows=[tuple(w) for w in x["windows"]]) for x in json.load(open(Path(__file__).with_name("treep_lines_extra.json"))) if x["id"] not in {l["id"] for l in TREEP_LINES})   # stop coordinates from the treep.app stop pages (treep.app/stops/35/<id>)

    def treep_anchors(key):
        out = []
        for _id, c in tstops[key]:
            c = tuple(c)
            if 20.40 <= c[0] <= 20.80 and -103.58 <= c[1] <= -103.00 and (not out or hav(c, out[-1]) > 150):
                out.append(c)
        return out
    specs = [dict(ln, anchors=treep_anchors(ln["id"]) if ln["id"] in tstops else resolve(ln["names"], ln["area"]), exact=ln["id"] in tstops) for ln in TREEP_LINES]
    specs.append({"id": "C72", "long": "C72 Atemajac del Valle - Zapopan - Cañón de las Flores", "run_min": 85, "anchors": C72_ANCHORS,
                  "windows": [(5 * 60, 23 * 60, 15), (5 * 60, 23 * 60, 15)]})
    for ln in specs:
        pts = ln["anchors"]
        if len(pts) < 2:
            print(f"{ln['id']}: skipped (fewer than 2 anchors located)"); continue
        shape, last = [], None
        for p_ in pts:
            nid = roads.nearest(p_)
            if last is not None and nid != last:
                seg = roads.path(last, nid)
                if seg is None:
                    print(f"    {ln['id']}: no road path to {p_}; anchor dropped"); continue
                seg_len = along(seg)[-1]
                if seg_len > 2.2 * hav(roads.xy[last], roads.xy[nid]) + 800:   # a mislocated anchor makes the road path double back: drop it
                    print(f"    {ln['id']}: anchor {p_} dropped (path {seg_len / 1000:.1f} km for a {hav(roads.xy[last], roads.xy[nid]) / 1000:.1f} km hop)"); continue
                shape.extend(seg)
            elif last is None:
                shape.append(roads.xy[nid])
            last = nid
        if len(shape) < 3:
            print(f"{ln['id']}: skipped (no path)"); continue
        cum = along(shape); L = cum[-1]
        straight = hav(pts[0], pts[-1])
        speed = L / 1000 / (ln["run_min"] / 60)
        flag = "" if 10 <= speed <= 40 and (L <= 5.5 * max(straight, 1) or (ln.get("exact") and straight < 1500)) else "  <-- CHECK path"
        speed = max(8.0, min(45.0, speed))
        picked = [("Terminal", shape[0][0], shape[0][1], 0.0)]
        if ln.get("exact"):
            ks, picked = 0, [("Terminal", shape[0][0], shape[0][1], 0.0)]
            for (la, lo) in pts[1:]:
                k = min(range(ks, len(shape)), key=lambda q: hav(shape[q], (la, lo)))
                ks = k
                if cum[k] - picked[-1][3] >= 150:
                    picked.append(("Stop", la, lo, cum[k]))
            near = []
        else:
            near = None
        near = near if near is not None else sorted(((cum[k], nm, la, lo) for (la, lo, nm) in osm_stops for k in [min(range(len(shape)), key=lambda q: hav(shape[q], (la, lo)))] if hav(shape[k], (la, lo)) <= 40), key=lambda r: r[0])
        for al, nm, la, lo in near:
            if al - picked[-1][3] >= 300:
                picked.append((nm, la, lo, al))
        if L - picked[-1][3] > 300:
            picked.append(("Terminal", shape[-1][0], shape[-1][1], L))
        filled = []
        for r_ in picked:
            if filled and r_[3] - filled[-1][3] > 1000:
                base = filled[-1][3]; gap = r_[3] - base; n_ = int(gap // 700)
                for q in range(1, n_ + 1):
                    al = base + q * gap / (n_ + 1); k = min(range(len(shape)), key=lambda z: abs(cum[z] - al))
                    filled.append(("Paradero (interpolated)", shape[k][0], shape[k][1], al))
            filled.append(r_)
        back = sorted([(n, la, lo, L - a) for n, la, lo, a in filled], key=lambda r: r[3])
        if flag:
            print(f"{ln['id']}: NOT written: the path does not match the timetable's run time or is far longer than the stops' span ({L / 1000:.1f} km, straight {straight / 1000:.1f} km, {speed:.0f} km/h)")
            continue
        routes.append({"id": "assumed_treep_" + norm(ln["id"]), "short": ln["id"], "long": ln["long"],
                       "desc": "ASSUMED: route listed on treep.app (Mi Transporte) but absent from the official GTFS; real weekday timetable (interval %d min); path on OSM roads through %d located stops of the route, OSM stops within 40 m" % (ln["windows"][0][2], len(pts)),
                       "shape": shape, "speed_kmh": speed, "directions": [("Outbound", filled), ("Inbound", back)],
                       "windows": [[ln["windows"][0]], [ln["windows"][1]]]})
        print(f"{ln['id']}: {len(pts)} anchors, {L / 1000:.1f} km (straight {straight / 1000:.1f}), {L / 1000 / (ln['run_min'] / 60):.0f} km/h implied by the {ln['run_min']} min run, {len(filled)} stops, every {ln['windows'][0][2]} min{flag}")
    md = ("# ASSUMED feed: Guadalajara lines from treep.app\n\nNot official data. Lines of the Mi Transporte network (treep.app, agency 35) that are not in the official feed (mdb-2366, calendar 2024) or in OSM: %s.\n"
          "STOPS: treep.app prints the stops of every trip (https://treep.app/index/trip/35/<route>/<trip>) and each stop page (https://treep.app/stops/35/<id>) carries its latitude/longitude; "
          "C72's stop ids encode them (jalisco_3_<lat>_W<lon>). These coordinates are the stops of the feed (stops closer than 150 m are merged); lines whose trip page lists no stops were placed from named stops geocoded on OSM.\n"
          "PATH: shortest path on the OSM road network through the stops in order; an anchor that makes the path double back (more than 2.2x the straight hop + 800 m) is dropped. A line is written only if the implied speed (path / printed run time) is 10-40 km/h; "
          "loops (ends within 1.5 km) are exempt from the path-vs-straight-line check.\n"
          "TIMES: the weekday timetable printed on the trip page: first departure, last departure, median gap between departures (the interval) and the median duration (run time); outbound and inbound are separate trips.\n"
          "NOT DUPLICATES: each line was compared with the official feed (share of its stops within 60 m of the stops of the best-matching official route: 33-55%% for C12, C58, C111-2, C96-V1, T06-2, T16B-C04, T16B-C05; C51-V1 69%% but the official feed has V2 only).\n"
          "Checked: stops are 12-17 m (median) from the drawn path on every line; 4 stops of 600 are 150-370 m off (C84, T22-V02).\n\n"
          "HEADWAY_BASIS: line-specific published timetables, not feed medians.\n") % ", ".join(r["short"] for r in routes)
    write_feed(ROOT / "guadalajara" / "gtfs" / "assumed_treep_lines", "treep.app Mi Transporte timetables + OpenStreetMap", "Mi Transporte lines from treep.app (assumed)", routes, md)



# ------------------------------------------------------------------ Guadalajara: Line 4 of the Tren Electrico Urbano (opened 2025-12-15, after the 2024 official feed)
L4_STATIONS = [("Las Juntas", 20.6079639, -103.3407694), ("Jalisco 200 Años", 20.5759889, -103.3552194), ("Real del Valle", 20.55889, -103.3659083),
               ("Concepción del Valle", 20.527139, -103.3849389), ("El Cuervo", 20.5076278, -103.3934083), ("Lomas del Sur", 20.4856472, -103.4050389),
               ("CUTLAJO", 20.4689583, -103.4152194), ("Tlajomulco Centro", 20.469389, -103.436583)]


def build_guadalajara_line4():
    """Line 4 (Las Juntas - Tlajomulco Centro): 8 open stations (Acueducto is not open yet), coordinates from the Wikipedia station table; the track is drawn station to station (no
    track geometry is available); 38 min end to end for 21.1 km (SITEUR) gives the speed; every 9 min (state communique / SITEUR); service 05:00-21:30 (the SITEUR page says 04:50-23:00, the communique 04:00-21:30: the overlap is used)."""
    pts = [(la, lo) for _, la, lo in L4_STATIONS]
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + hav(a, b))
    L = cum[-1]
    speed = 21.1 / (38 / 60)   # km/h from the official figures (the polyline is ~%.1f km)
    out = [(n, la, lo, c) for (n, la, lo), c in zip(L4_STATIONS, cum)]
    back = [(n, la, lo, L - c) for (n, la, lo, c) in reversed(out)]
    r = {"id": "assumed_line4_tren", "short": "L4", "long": "Línea 4 Tren Eléctrico Urbano: Las Juntas - Tlajomulco Centro", "type": 0,
         "desc": "ASSUMED: Line 4 opened 2025-12-15 (after the 2024 official feed); 8 stations with Wikipedia coordinates, track drawn station to station; every 9 min (SITEUR), 05:00-21:30",
         "shape": pts, "speed_kmh": speed * L / 21100 if L else speed, "directions": [("Tlajomulco Centro", out), ("Las Juntas", back)],
         "windows": [[(5 * 60, 21 * 60 + 30, 9)], [(5 * 60, 21 * 60 + 30, 9)]]}
    md = ("# ASSUMED feed: Guadalajara Line 4 (Tren Eléctrico Urbano)\n\nNot official data. Line 4 opened on 2025-12-15, after the calendar-2024 official feed (mdb-2366), which has an older planned 'L4' (Juan Gil Preciado - Valle de los Molinos) instead.\n"
          "STATIONS: Las Juntas, Jalisco 200 Años, Real del Valle, Concepción del Valle, El Cuervo, Lomas del Sur, CUTLAJO, Tlajomulco Centro (Acueducto is not open); coordinates from the Wikipedia station table (es.wikipedia.org/wiki/Línea_4_del_Tren_Eléctrico_Urbano_de_Guadalajara).\n"
          "TRACK: straight segments between stations (21.1 km, 38 min end to end according to SITEUR/state communique; the drawn length is %.1f km).\n"
          "HEADWAY_BASIS: line-specific: every 9 min (state communique and SITEUR, December 2025); hours 05:00-21:30 (SITEUR page 04:50-23:00; communique 04:00-21:30; the overlap is used). route_type 0 (tram/light rail).\n"
          "Sources: https://www.siteur.gob.mx/index.php/noticias/item/puesta-en-operacion-de-la-linea-4-del-tren-ligero-conectara-tlajomulco-con-la-metropoli-en-solo-38-minutos , https://www.siteur.gob.mx/index.php/sistemas-de-transporte/mi-tren , https://www.infobae.com/mexico/2025/12/16/abre-la-linea-4-del-tren-ligero-en-jalisco-con-senalamientos-por-obras-sin-terminar/\n") % (L / 1000)
    write_feed(ROOT / "guadalajara" / "gtfs" / "assumed_line4_tren", "SITEUR + Wikipedia station coordinates", "Guadalajara Line 4 Tren Eléctrico Urbano (assumed)", [r], md)


# ------------------------------------------------------------------ Guadalajara: Line 5 Mi Macro Aeropuerto (opened 2026-06-04, after the 2024 official feed)
L5_NAMES = ["Chapalita Inn", "El Colli", "Felipe Ruvalcaba", "Miramar", "Mariano Otero", "El Briseño", "Agrícola", "López Mateos", "ITESO", "Terminal Sur de Autobuses", "Periférico Sur",
            "San Sebastianito", "8 de Julio", "Toluquilla", "Adolf Horn", "Jalisco 200 Años", "Artesanos", "Las Pintas", "Carretera a Chapala", "Las Liebres", "Parque Montenegro",
            "Las Torres", "Las Pintitas", "La Gigantera", "La Piedrera", "San José del Quince", "Aeropuerto"]
L5_FIXED = {"Jalisco 200 Años": (20.5759889, -103.3552194), "Aeropuerto": (20.5218, -103.3111)}


#: Line 5 stations whose coordinates are known: stops with exactly that name in the official feed (Mi Macro Periférico stations, mdb-2366), the Line 4 table (Jalisco 200 Años), OSM ("Toluquilla", "Artesanos" on Av. de los Artesanos) and the airport terminal
L5_KNOWN = {"Chapalita Inn": (20.654902, -103.44612), "Felipe Ruvalcaba": (20.642962, -103.44114), "Miramar": (20.635942, -103.4382), "El Briseño": (20.625244, -103.43359),
            "ITESO": (20.6101, -103.4155), "Terminal Sur de Autobuses": (20.60852, -103.40752), "Periférico Sur": (20.607162, -103.401054), "San Sebastianito": (20.590712, -103.38588),
            "Toluquilla": (20.57953, -103.36921), "Adolf Horn": (20.577204, -103.360695), "Jalisco 200 Años": (20.5759889, -103.3552194), "Artesanos": (20.58207, -103.351),
            "Las Pintas": (20.587606, -103.32689), "Aeropuerto": (20.5218, -103.3111)}


def build_guadalajara_line5():
    """Line 5 Mi Macro Aeropuerto, 27 published stations, 32 km. 14 stations have known coordinates (L5_KNOWN); the ones between Las Pintas and the airport (Carretera a Chapala, Las Liebres, Parque Montenegro,
    Las Torres, Las Pintitas, La Gigantera, La Piedrera, San José del Quince) and the unlocated ones elsewhere (El Colli, Mariano Otero, Agrícola, López Mateos, 8 de Julio) keep their names and are spaced evenly along the OSM
    road path between their known neighbours. Headway from the 41-bus fleet (fleet method), hours 03:30-22:30 (SITEUR)."""
    bbox = (20.40, -103.58, 20.80, -103.00)
    roads = Roads("gdl_treep", bbox)
    known = [(i, L5_KNOWN[n]) for i, n in enumerate(L5_NAMES) if n in L5_KNOWN]
    shape, last, idx_shape = [], None, {}
    for i, p_ in known:
        nid = roads.nearest(p_)
        if last is not None and nid != last:
            seg = roads.path(last, nid)
            if seg is None or along(seg)[-1] > 2.5 * hav(roads.xy[last], roads.xy[nid]) + 800:
                print(f"    L5: anchor {L5_NAMES[i]} dropped"); continue
            shape.extend(seg)
        elif last is None:
            shape.append(roads.xy[nid])
        idx_shape[i] = len(shape) - 1
        last = nid
    cum = along(shape); L = cum[-1]
    print(f"L5: {len(idx_shape)} anchors kept, path {L / 1000:.1f} km (published 32 km)")
    if not 24000 <= L <= 40000:
        print("L5: NOT written, the path length does not match the published 32 km"); return
    kept = sorted(idx_shape)
    pos = {}
    for i in kept:
        pos[i] = cum[idx_shape[i]]
    for a_, b_ in zip(kept, kept[1:]):          # stations between two known neighbours: evenly spaced along the path between them
        gap = [j for j in range(a_ + 1, b_)]
        for q, j in enumerate(gap, 1):
            pos[j] = pos[a_] + q * (pos[b_] - pos[a_]) / (len(gap) + 1)
    stations = []
    for i, nm in enumerate(L5_NAMES):
        if i not in pos:
            continue
        k = min(range(len(shape)), key=lambda q: abs(cum[q] - pos[i]))
        stations.append((nm + ("" if i in idx_shape else " (interpolated)"), shape[k][0], shape[k][1], cum[k]))
    speed = 25.0
    hw = fleet_headway(41, 2 * L / 1000, speed)
    back = sorted([(nm, la, lo, L - a) for nm, la, lo, a in stations], key=lambda r: r[3])
    r = {"id": "assumed_line5_macro", "short": "L5", "long": "Línea 5 Mi Macro Aeropuerto: Chapalita Inn - Aeropuerto", "type": 3,
         "desc": f"ASSUMED: Line 5 opened 2026-06-04 (after the 2024 official feed); {len(idx_shape)} of 27 stations at known coordinates, the others spaced along the OSM road path; headway {hw} min from the 41-bus fleet, 03:30-22:30",
         "shape": shape, "speed_kmh": speed, "directions": [("Aeropuerto", stations), ("Chapalita Inn", back)], "windows": [[(3 * 60 + 30, 22 * 60 + 30, hw)], [(3 * 60 + 30, 22 * 60 + 30, hw)]]}
    md = ("# ASSUMED feed: Guadalajara Line 5 Mi Macro Aeropuerto\n\nNot official data. Line 5 opened on 2026-06-04, after the calendar-2024 official feed (mdb-2366).\n"
          "PUBLISHED: 27 stations Chapalita Inn - Aeropuerto, 32 km, 41 electric buses, 03:30-22:30 from 2026-06-08 (Wikipedia 'Guadalajara Mi Macro'; Jalisco government release 2026-06-05; https://linea5.jalisco.gob.mx/; SITEUR). No headway is published.\n"
          "STATIONS: %d have known coordinates: the Mi Macro Periférico stations with the same exact name in the official feed (Chapalita Inn, Felipe Ruvalcaba, Miramar, El Briseño, Terminal Sur de Autobuses, San Sebastianito, Adolf Horn, Las Pintas), "
          "ITESO and Periférico Sur (official feed / OSM), Toluquilla and Artesanos (OSM), Jalisco 200 Años (Line 4 station table), the airport terminal. The rest (El Colli, Mariano Otero, Agrícola, López Mateos, 8 de Julio, Carretera a Chapala, Las Liebres, Parque Montenegro, Las Torres, Las Pintitas, La Gigantera, La Piedrera, San José del Quince) "
          "keep their published order and are spaced evenly along the road path between their known neighbours (marked 'interpolated'); no source with their coordinates was found.\n"
          "PATH: shortest OSM road path through the known stations (%.1f km here, 32 km published).\n"
          "HEADWAY_BASIS: fleet method: round trip 2 x %.1f km at %d km/h with 10%% layover, buses in service 41 x (1 - 8%% workshop - 9%% spare) -> every %d min on the trunk (three extra service patterns share the fleet).\n") % (len(idx_shape), L / 1000, L / 1000, speed, hw)
    write_feed(ROOT / "guadalajara" / "gtfs" / "assumed_line5_macro", "SITEUR + Wikipedia station list + official feed stops + OpenStreetMap", "Guadalajara Line 5 Mi Macro Aeropuerto (assumed)", [r], md)


# ------------------------------------------------------------------ Shanghai metro area: intercity rail (China Railway services CPTOND does not contain)
SH_RAIL_STATIONS = {   # OpenStreetMap railway=station nodes (name -> first match), queried by `build_shanghai_intercity_rail`
    "huning": ["上海虹桥", "安亭北", "昆山", "阳澄湖", "苏州园区", "苏州", "苏州新区", "无锡新区", "无锡", "惠山", "戚墅堰", "常州"],
    "jinghu": ["上海虹桥", "昆山南", "苏州北", "无锡东", "常州北"],
    "huning_conv": ["昆山", "苏州", "无锡", "常州"],
    "hukun_conv": ["上海松江", "嘉善", "嘉兴", "桐乡", "海宁"],
    "songjiang_shuttle": ["上海松江", "上海南"],
    "xuanhang_conv": ["安吉", "长兴", "湖州", "德清"],
    "ninghang": ["溧阳", "宜兴", "湖州", "德清"],
    "xiaoyong_conv": ["宁波", "余姚"],
    "hangyong": ["宁波", "余姚北"],
    "yongtai": ["宁波", "奉化", "宁海"],
    "ningqi": ["如皋", "白蒲", "南通"],
    "ningqi_east": ["南通", "海门", "启东"],
    "husutong": ["上海虹桥", "安亭西", "太仓", "常熟", "张家港", "南通西"],
    "huhang": ["上海虹桥", "金山北", "嘉善南", "嘉兴南", "海宁西"],
    "huhu": ["上海虹桥", "上海松江", "苏州南", "盛泽", "湖州南浔", "湖州东"],
    "jinshan": ["上海南", "莘庄", "春申", "车墩", "亭林", "金山园区", "金山卫"],
}


def build_shanghai_intercity_rail():
    names = sorted({n for v in SH_RAIL_STATIONS.values() for n in v})
    q = '[out:json][timeout:200];node["railway"~"^(station|halt)$"]["name"~"^(' + "|".join(names) + ')站?$"](29.2,119.4,32.4,122.6);out tags center 3000;'
    d = overpass("sh_rail_stations_" + hashlib.md5(q.encode()).hexdigest()[:8], q)
    pos = {}
    for e in d["elements"]:
        n = e["tags"]["name"].rstrip("站")
        if n == "莘庄" and e["tags"].get("network") == "上海地铁":
            continue   # the national-railway Xinzhuang node is the Jinshan line's; the metro node is a different one
        pos.setdefault(n, (e["lat"], e["lon"]))
    peak = lambda a, b_: [(345, 420, a), (420, 540, b_), (540, 1020, a), (1020, 1140, b_), (1140, 1380, a)]
    spec = [   # key, short, long, speed km/h, windows, desc
        ("huning", "Huning", "Shanghai Hongqiao - Suzhou - Wuxi - Changzhou (Shanghai-Nanjing intercity railway, China Railway)", 120, peak(8, 5),
         "ASSUMED: stations from OpenStreetMap, straight track between stations, 120 km/h average incl. stops; 5 min at peak (minimum headway 5 min) and 8 min otherwise, 05:45-23:00 (110-142 pairs a day on the intercity line, Pengpai / Changzhou government / Jiangsu news; Suzhou station alone has 192 train pairs a day in the Jan 2026 plan, Wikipedia)"),
        ("jinghu", "Jinghu HSR", "Shanghai Hongqiao - Kunshan South - Suzhou North - Wuxi East - Changzhou North (Beijing-Shanghai high-speed railway)", 160, [(360, 1320, 11)],
         "ASSUMED: stations from OpenStreetMap, straight track, 160 km/h average incl. stops; every 11 min 06:00-22:00 from about 98.5 pairs a day on the Shanghai-Nanjing section (one self-media article; LOW CONFIDENCE, trains skip some of these stations)"),
        ("huning_conv", "Hu-Ning conventional", "Kunshan - Suzhou - Wuxi - Changzhou (conventional Beijing-Shanghai railway, Shanghai-Nanjing section: K/T/Z and long-distance trains)", 100, [(360, 1320, 18)],
         "ASSUMED: stations from OpenStreetMap, straight track, 100 km/h average incl. stops; every 18 min 06:00-22:00 from the 57 pairs of conventional passenger trains a day on the Hu-Ning section of the Jinghu railway (Wikipedia Suzhou station, Jan 2026 plan; 10 EMU pairs besides); most of them stop only at the big stations (MEDIUM-LOW CONFIDENCE); Shanghai South's conventional trains moved to Songjiang station on 2025-07-01 and are not modelled"),
        ("hukun_conv", "Hu-Kun conventional", "Songjiang - Jiashan - Jiaxing - Tongxiang - Haining (conventional Shanghai-Hangzhou railway: K/T/Z and long-distance trains)", 90, [(360, 1320, 20)],
         "ASSUMED: stations from OpenStreetMap, straight track, 90 km/h average incl. stops; every 20 min 06:00-22:00: Songjiang station handles 77 conventional trains a day (official, Q1 2025; it took over Shanghai South's conventional trains on 2025-07-01) and Jiaxing station 154 trains a day (railway timetable site); ESTIMATE, most long-distance trains stop only at the big stations (LOW CONFIDENCE)"),
        ("songjiang_shuttle", "Songjiang-Shanghai South shuttle", "Songjiang - Shanghai South (C500 / C3046 commuter trains added 2025-07-01)", 75,
         ([(405, 495, 70)], [(1050, 1170, 60)]),
         "ASSUMED: stations from OpenStreetMap; C500 leaves Songjiang 06:48 (20 min) and C3046 07:59 (16 min) towards Shanghai South (Shanghai Bendibao timetable notice); the two morning trains are modelled as one every 70 min 06:45-08:15; the return trains are NOT published in what I found: one every 60 min 17:30-19:30 is an ESTIMATE (LOW CONFIDENCE); 75 km/h (about 25 km in 20 min)"),
        ("xuanhang_conv", "Xuan-Hang / Ning-Hang conventional", "Anji - Changxing - Huzhou - Deqing (conventional Xuancheng-Hangzhou railway, the route of the conventional Nanjing-Hangzhou trains such as T151 and the Hangzhou-Xining K trains)", 80, [(360, 1260, 120)],
         "ASSUMED: stations from OpenStreetMap; every 120 min 06:00-21:00 ESTIMATE from about 7 pairs a day: Guangde station, on the same line in Anhui, has 14 stopping trains a day (railway timetable site), an old travel page says 6 trains a day pass Huzhou, and the stopping counts of 87 at Anji, 81 at Deqing and 58 at Changxing South are mostly high-speed trains of other lines; the conventional Nanjing-Hangzhou trains run over this line (no separate Ning-Hang conventional line exists in the area); LOW CONFIDENCE"),
        ("ninghang", "Ning-Hang HSR", "Liyang - Yixing - Huzhou - Deqing (Nanjing-Hangzhou high-speed railway)", 200, [(390, 1320, 15)],
         "ASSUMED: stations from OpenStreetMap, straight track; every 15 min 06:30-22:00 ESTIMATE: Huzhou station has 288 stopping trains a day (railway timetable site) shared with other services, Deqing 81 and Changxing South 58 (the line's stations are served by only part of the trains); LOW CONFIDENCE"),
        ("xiaoyong_conv", "Xiao-Yong conventional", "Ningbo - Yuyao (conventional Xiaoshan-Ningbo railway)", 80, [(360, 1260, 60)],
         "ASSUMED: stations from OpenStreetMap; every 60 min 06:00-21:00 ESTIMATE: Yuyao station had 16 conventional pairs a day, all through trains from Ningbo (Wikipedia, June 2017) and 62 trains a day stop there now, of which 9 pairs are the S1 intercity and the Hangzhou-Ningbo C636; LOW CONFIDENCE"),
        ("hangyong", "Hangzhou-Ningbo HSR", "Ningbo - Yuyao North (Hangzhou-Ningbo high-speed railway, towards Hangzhou East)", 180, [(360, 1350, 7)],
         "ASSUMED: stations from OpenStreetMap, straight track, 180 km/h average incl. stops; every 7 min 06:00-22:30: Ningbo station has 416 stopping trains a day (about 208 a direction, railway timetable site) and 103-118 direct trains run Ningbo-Hangzhou East (Qunar / Ctrip snapshots, 2026; fastest 41-54 min) (MEDIUM-LOW CONFIDENCE: Yuyao North is served by only some trains)"),
        ("yongtai", "Yong-Tai-Wen HSR", "Ningbo - Fenghua - Ninghai (Ningbo-Taizhou-Wenzhou high-speed railway)", 130, [(380, 1320, 15)],
         "ASSUMED: stations from OpenStreetMap, straight track, 130 km/h average incl. stops (Ninghai-Ningbo about 30 min); every 15 min 06:20-22:00 as the rest of the 416 trains a day at Ningbo station (no count for this line; LOW CONFIDENCE)"),
        ("ningqi", "Nanjing-Qidong railway (Nantong)", "Rugao - Baipu - Nantong (Nanjing-Qidong railway, CR200J and EMU trains)", 110, [(420, 1320, 12)],
         "ASSUMED: stations from OpenStreetMap, straight track, 110 km/h average incl. stops; every 12 min 07:00-22:00: Nantong station has 212 stopping trains a day (about 106 a direction, railway timetable site) and the Nanjing-Qidong line had 41.5 pairs from 2019-10 (Xinhua Daily) (LOW CONFIDENCE, only part of the trains stop at Baipu)"),
        ("ningqi_east", "Nanjing-Qidong railway (Qidong)", "Nantong - Haimen - Qidong (Nanjing-Qidong railway)", 110, [(420, 1320, 90)],
         "ASSUMED: stations from OpenStreetMap, straight track, 110 km/h average; every 90 min 07:00-22:00 from the 6 pairs a day on Qidong-Nantong (Xinhua Daily 2019; LOW CONFIDENCE)"),
        ("husutong", "Hu-Su-Tong", "Shanghai Hongqiao - Taicang - Changshu - Zhangjiagang - Nantong West (Shanghai-Suzhou-Nantong railway)", 130, [(360, 1350, 25)],
         "ASSUMED: stations from OpenStreetMap, straight track, 130 km/h average incl. stops; every 25 min 06:00-22:30 (37-43 pairs a day at Taicang, Changshu and Zhangjiagang at opening in 2020, Bendibao / SASAC)"),
        ("huhu", "Hu-Su-Hu HSR", "Shanghai Hongqiao - Songjiang - Suzhou South - Shengze - Nanxun - Huzhou East (Shanghai-Suzhou-Huzhou high-speed railway)", 200, [(360, 1320, 15)],
         "ASSUMED: stations from OpenStreetMap, straight track, 200 km/h average incl. stops (Hongqiao-Huzhou 55 min, Songjiang-Huzhou East 37 min, Hongqiao-Nanxun 43 min in the opening reports); every 15 min 06:00-22:00: 38 trains a day at opening in Dec 2024, 229 a day network-wide from Q1 2025, and 127 trains a day between Nanxun and Hangzhou West on 2026-11-02 (Qunar) (LOW CONFIDENCE: no per-station count for the Shanghai side)"),
        ("huhang", "Huhang", "Shanghai Hongqiao - Jinshan North - Jiashan South - Jiaxing South - Haining West (Shanghai-Hangzhou high-speed railway)", 150, [(360, 1350, 7)],
         "ASSUMED: stations from OpenStreetMap (Songjiang South not located), straight track, 150 km/h average incl. stops; every 7 min 06:00-22:30 (Jiaxing South has 299 stopping trains a day, i.e. about 150 a direction, railway timetable site; timetable sites list 168-178 trains between Hongqiao and Hangzhou East; the earlier 13 min came from the 80 pairs of the 2010 plan; LOW CONFIDENCE: no per-station count found)"),
        ("jinshan", "Jinshan", "Shanghai South - Jinshanwei (Jinshan railway, Shanghai suburban railway)", 65, [(330, 1320, 15)],
         "ASSUMED: stations from OpenStreetMap, straight track, 65 km/h average incl. stops; every 15 min 05:30-22:00 (10-20 min, 47 trains a day, first 05:30 and last 21:57; Baidu Zhidao / Sohu 2025 timetable)"),
    ]
    routes = []
    for key, short, long_, speed, win, desc in spec:
        pts = [(n, *pos[n]) for n in SH_RAIL_STATIONS[key] if n in pos]
        miss = [n for n in SH_RAIL_STATIONS[key] if n not in pos]
        if miss:
            print(f"    {key}: stations not found in OSM: {miss}")
        xy = [(la, lo) for _, la, lo in pts]
        cum = along(xy)
        fwd = [(n + "站", la, lo, c) for (n, la, lo), c in zip(pts, cum)]
        back = [(n, la, lo, cum[-1] - c) for (n, la, lo, c) in reversed(fwd)]
        routes.append({"id": f"assumed_rail_{key}", "short": short, "long": long_, "desc": desc, "shape": xy, "speed_kmh": speed, "type": 2,
                       "directions": [(fwd[-1][0], fwd), (fwd[0][0], back)], "windows": list(win) if isinstance(win, tuple) else [win, win]})
    md = ("# ASSUMED feed: intercity and suburban rail of the Shanghai metro area\n\nNot official data. China Railway services that the CPTOND dataset does not contain "
          "(it holds only Ningbo S1, kept in the main feed).\n\n" + "\n".join(f"* **{r['short']}**: {r['desc']}" for r in routes) +
          "\n\nNot covered: the Xin-Chang railway (Wikipedia: only Xinyi-Haian carries passenger trains, none in this area) and conventional trains of lines without a count (Ning-Hang conventional), the Hangzhou-Huzhou part of Hu-Su-Hu beyond Huzhou East, the trains of Shanghai Station, Shanghai East and Songjiang South "
          "(not found in OpenStreetMap), and the Hangzhou-Haining intercity line. Stops are placed at the OSM station nodes; trains are "
          "modelled as one frequency-based service per corridor.\n")
    write_feed(ROOT / "shanghai" / "gtfs" / "assumed_intercity_rail", "OpenStreetMap + news and Baidu sources, see ASSUMED.md", "Shanghai metro area intercity rail (assumed)", routes, md)


# ------------------------------------------------------------------ Shanghai metro area: rapid transit missing from CPTOND (found with an OpenStreetMap census of all rail stations)
def _osm_rel_stops(rel_id, cache_name, names_pool):
    """Ordered (name, lat, lon) stops of an OSM route relation (stop nodes, else platform nodes without duplicates); unnamed ones get the nearest named station within 300 m."""
    d = overpass(cache_name, "")
    el = {(e["type"], e["id"]): e for e in d["elements"]}
    rel = el[("relation", rel_id)]
    roles = ("stop", "stop_entry_only", "stop_exit_only")
    nodes = [el[(m["type"], m["ref"])] for m in rel["members"] if m["type"] == "node" and m["role"] in roles and ("node", m["ref"]) in el]
    plats = [el[(m["type"], m["ref"])] for m in rel["members"] if m["type"] == "node" and m["role"] in ("platform", "platform_entry_only", "platform_exit_only") and ("node", m["ref"]) in el]
    uniq = lambda L: len({n.get("tags", {}).get("name", n["id"]) for n in L})
    if len(nodes) < 2 or uniq(plats) > uniq(nodes):   # tram relations carry few stop nodes but one named platform per stop
        nodes = plats
    out, seen = [], set()
    for n in nodes:
        nm = n.get("tags", {}).get("name")
        if not nm:
            best = min(names_pool, key=lambda q: hav((n["lat"], n["lon"]), (q[1], q[2])), default=None)
            nm = best[0] if best and hav((n["lat"], n["lon"]), (best[1], best[2])) < 300 else f"Stop {len(out) + 1}"
        if nm in seen or any(hav((n["lat"], n["lon"]), (q[1], q[2])) < 150 for q in out):   # one platform per direction: keep the first of each stop
            continue
        seen.add(nm); out.append((nm, n["lat"], n["lon"]))
    return out


def build_shanghai_osm_rapid_transit():
    pool = [(e["tags"]["name"], e["lat"], e["lon"]) for e in json.loads((ROOT.parent / "cache" / "osm" / "shanghai_rail" / "all_rail_stations.json").read_text()) if e.get("tags", {}).get("name")]
    peak = lambda a, b_, c: [(360, 420, c), (420, 540, a), (540, 1020, b_), (1020, 1140, a), (1140, 1260, b_), (1260, 1320, c)]
    spec = [   # key, short, long, relation id, cache, type, speed km/h, windows, desc
        ("nb6", "Ningbo 6", "Ningbo Rail Transit Line 6 (Gulin - Honglian)", 20085685, "sh_rail_rel_members_v3", 1, 38, peak(6, 8, 10),
         "ASSUMED: stops from OpenStreetMap; opened 2026-01-16 (phase 1); 6 min peak, 8 min off-peak, 10 min low, 06:00-22:00 (Ningbo Rail Transit opening notice); 38 km/h commercial speed assumed (design 80 km/h)"),
        ("nb7", "Ningbo 7", "Ningbo Rail Transit Line 7 (Yunlong - Yufan)", 19533620, "sh_rail_rel_members_v3", 1, 40, peak(6, 8, 10),
         "ASSUMED: stops from OpenStreetMap; opened 2025-08-29, GoA4, max 100 km/h; 6 min peak, 8 min off-peak, 10 min low, 06:00-22:00 (CREC / Ningbo notice); 40 km/h commercial speed assumed"),
        ("nb8", "Ningbo 8", "Ningbo Rail Transit Line 8 phase 1 (Hansonglu - Kaiyuanlu)", 19314268, "sh_rail_rel_members_v3", 1, 38, peak(6, 8, 10),
         "ASSUMED: stops from OpenStreetMap; opened 2025-06-30; 6 min peak, 8 min off-peak, 10 min low, 06:00-22:00 (Ningbo government); 38 km/h commercial speed assumed (max 80-100 km/h)"),
        ("nb12", "Ningbo 12", "Ningbo Rail Transit Line 12 (Xiangshan - Xiaoyangjiang)", 21385756, "sh_rail_rel_members_v3", 2, 72, [(390, 420, 20), (420, 540, 12), (540, 1020, 20), (1020, 1140, 12), (1140, 1275, 20)],
         "ASSUMED: stops from OpenStreetMap; opened 2026-09-22 (first cross-sea suburban line, 61.45 km, 10 stations, 4-car type A, max 160 km/h; first/last trains 06:30-21:00/21:20 in the trial notice); NO headway published: ESTIMATE 12 min peak / 20 min off-peak (a 4-car suburban line serving Xiangshan, about 0.6M people); 72 km/h commercial speed estimated (stations 6.8 km apart, ~49 min end to end)"),
        ("nb12x", "Ningbo 12 Express", "Ningbo Rail Transit Line 12 weekday peak express (Xiaoyangjiang - Dancheng - Damuwan)", 21385756, "sh_rail_rel_members_v3", 2, 116, [(420, 540, 30), (1020, 1140, 30)],
         "ASSUMED: peak-only express that stops only at Danzheng in between and takes about 30 min end to end (Tencent News trial report, Sep 2026); every 30 min in the two peaks is an ESTIMATE"),
        ("chuansha", "Chuansha Express", "Shanghai Chuansha Express (Chuansha Economic Park - Huadai Road Chuanhuan South Road)", 18331525, "sh_chuansha_rel_v1", 3, 24, [(330, 420, 20), (420, 540, 12), (540, 1020, 20), (1020, 1140, 12), (1140, 1320, 20)],
         "ASSUMED: stops from OpenStreetMap; NO timetable found: ESTIMATE of a Pudong district trunk express like the Haiwan Express (peak 12 min, off-peak 20 min, 05:30-22:00, 24 km/h); LOW CONFIDENCE"),
        ("airport", "Airport Link", "Shanghai Airport Link (Pudong T1/T2 - Hongqiao T2)", 18468815, "sh_rail_rel_members_v3", 2, 88, [(360, 1335, 15)],
         "ASSUMED: stops from OpenStreetMap; opened 2024-12-27, single service, every 15 min 06:00-22:15 (Shanghai government / airport authority), 59 km in about 40 min = 88 km/h (design 160 km/h)"),
        ("apm_e", "Pudong APM East", "Pudong Airport passenger shuttle East line (T2 - S2)", 14660488, "sh_rail_rel_members_v3", 1, 30, [(330, 1410, 5)],
         "ASSUMED: stops from OpenStreetMap; opened 2019-09-16; every 5 min (reports say 5 min, one 2023 report 10), 2 min 30 s one way, max 80 km/h; hours assumed 05:30-23:30"),
        ("apm_w", "Pudong APM West", "Pudong Airport passenger shuttle West line (T1 - S1)", 14660487, "sh_rail_rel_members_v3", 1, 30, [(330, 1410, 5)],
         "ASSUMED: stops from OpenStreetMap; opened 2019-09-16; every 5 min (reports say 5 min, one 2023 report 10), 2 min 30 s one way, max 80 km/h; hours assumed 05:30-23:30"),
        ("fengpu", "Fengpu Express (BRT)", "Shanghai Fengpu Express BRT (Nanqiao Bus Station - Shendu Road)", 15904013, "sh_brt_rel_members_v1", 3, 27,
         [(315, 420, 11.5), (420, 540, 6.5), (540, 1020, 11.5), (1020, 1140, 6.5), (1140, 1350, 11.5)],
         "ASSUMED: stops from OpenStreetMap (12 stops, 20.6 km); Shanghai's first BRT, opened 2018-04-20; peak 5-8 min, off-peak 8-15 min, 05:15-22:30 (Wenhui / Bendibao / Wikipedia), 45 min one way = 27 km/h (reported average 25 km/h); dedicated lanes 05:00-24:00; not in CPTOND"),
        ("haiwan", "Haiwan Express", "Shanghai Haiwan Express (Haiwan Road Jinhuitang Road - Nanqiao Bus Station)", 18320922, "sh_brt_rel_members_v1", 3, 28,
         [(320, 420, 20), (420, 540, 12.5), (540, 1020, 20), (1020, 1140, 12.5), (1140, 1260, 20)],
         "ASSUMED: stops from OpenStreetMap (14.3 km); opened 2023-12-26; peak 10-15 min, off-peak 20 min, 05:20-21:00 (Fengxian government notice, since reported extended to 23:40 at Nanqiao), about 30 min one way = 28 km/h; not in CPTOND"),
        ("haining", "Hangzhou-Haining", "Hangzhou-Haining intercity railway (Linping South - Zhejiang University International Campus)", 13078548, "sh_rail_rel_members_v2", 1, 55, [(390, 1350, 10)],
         "ASSUMED: stops from OpenStreetMap; opened 2021-06-28; every 10 min 06:30-22:30 (Hangzhou Bendibao / Tencent), 6 pairs of express a day not modelled; 46 km, design speed 120 km/h, 55 km/h commercial speed assumed"),
        ("sj_t1", "Songjiang T1", "Songjiang Tram Line 1 (Canghua Road - Xinqiao Railway Station)", 14251681, "sh_rail_rel_members_v2", 0, 20, [(360, 420, 16), (420, 540, 13), (540, 1020, 16), (1020, 1140, 13), (1140, 1320, 16)],
         "ASSUMED: stops from OpenStreetMap; weekday 13 min peak and 16 min off-peak since 2025-05-01 (Sina / Songjiang tram notice); hours 06:00-22:00 assumed; 20 km/h assumed for a tram"),
        ("sj_t2a", "Songjiang T2 (Longma)", "Songjiang Tram Line 2 loop via Longma Road", 16159672, "sh_rail_rel_members_v2", 0, 20, [(360, 420, 16), (420, 540, 13), (540, 1020, 16), (1020, 1140, 13), (1140, 1320, 16)],
         "ASSUMED: stops from OpenStreetMap; weekday 13 min peak and 16 min off-peak since 2025-05-01; hours 06:00-22:00 assumed; 20 km/h assumed for a tram"),
        ("sj_t2b", "Songjiang T2 (Gufulin)", "Songjiang Tram Line 2 loop via Guangfulin Road", 16159673, "sh_rail_rel_members_v2", 0, 20, [(360, 420, 16), (420, 540, 13), (540, 1020, 16), (1020, 1140, 13), (1140, 1320, 16)],
         "ASSUMED: stops from OpenStreetMap; weekday 13 min peak and 16 min off-peak since 2025-05-01; hours 06:00-22:00 assumed; 20 km/h assumed for a tram"),
    ]
    routes = []
    for key, short, long_, rid, cache, rtype, speed, win, desc in spec:
        pts = _osm_rel_stops(rid, cache, pool)
        if key == "nb12x":   # the peak express stops only at Dancheng between the terminals
            pts = [q for i, q in enumerate(pts) if i in (0, len(pts) - 1) or q[0] == "丹城"]
        xy = [(la, lo) for _, la, lo in pts]
        cum = along(xy)
        fwd = [(n, la, lo, c) for (n, la, lo), c in zip(pts, cum)]
        back = [(n, la, lo, cum[-1] - c) for (n, la, lo, c) in reversed(fwd)]
        routes.append({"id": f"assumed_rt_{key}", "short": short, "long": long_, "desc": desc, "shape": xy, "speed_kmh": speed, "type": rtype,
                       "directions": [(fwd[-1][0], fwd), (fwd[0][0], back)], "windows": [win, win]})
    # Jiaxing tram T1: the OSM relation has no stop members; the line's tram_stop nodes (one per distinct name), chained from one end by nearest neighbour
    seen, cand = set(), []
    for e in json.loads((ROOT.parent / "cache" / "osm" / "shanghai_rail" / "all_rail_stations.json").read_text()):
        t_ = e["tags"]
        if t_.get("railway") == "tram_stop" and 30.6 < e["lat"] < 30.85 and 120.6 < e["lon"] < 120.9 and t_.get("name") and t_["name"] not in seen:
            seen.add(t_["name"]); cand.append((t_["name"], e["lat"], e["lon"]))
    if len(cand) >= 4:
        cur = min(cand, key=lambda c: c[1]); chain = [cur]; left = [c for c in cand if c is not cur]   # start at the southern end (Jiaxing South Station)
        while left:
            nxt = min(left, key=lambda c: hav((c[1], c[2]), (chain[-1][1], chain[-1][2]))); chain.append(nxt); left.remove(nxt)
        xy = [(c[1], c[2]) for c in chain]; cum = along(xy)
        fwd = [(c[0], c[1], c[2], a) for c, a in zip(chain, cum)]; back = [(n, la, lo, cum[-1] - a) for (n, la, lo, a) in reversed(fwd)]
        win = [(330, 420, 10.9), (420, 540, 7.9), (540, 1020, 10.9), (1020, 1140, 7.9), (1140, 1260, 10.9), (1260, 1380, 10)]
        routes.append({"id": "assumed_rt_jx_t1", "short": "Jiaxing T1", "long": "Jiaxing Tram T1 (Jiaxing South Station - Jiaxing Railway Station)",
                       "desc": "ASSUMED: stops = the OpenStreetMap tram stops of the line (12 of the real 16), chained from the south end; weekday peak 7 min 56 s, off-peak 10 min 52 s, 21:00-22:00 10 min (Jiaxing Bendibao 2024), hours 05:30-23:00; about 30 min end to end (operator 2022), 24 km/h assumed",
                       "shape": xy, "speed_kmh": 24, "type": 0, "directions": [(fwd[-1][0], fwd), (fwd[0][0], back)], "windows": [win, win]})
        print("    Jiaxing T1 stops:", len(chain))
    md = ("# ASSUMED feed: rapid transit missing from the CPTOND dataset (Shanghai metro area)\n\nNot official data. Found by comparing every OpenStreetMap rail station and route relation of the area with the main feed "
          "(`shanghai_metro_area`); each service below is listed with the source of its stops, headway and speed.\n\n" + "\n".join(f"* **{r['short']}**: {r['desc']}" for r in routes) +
          "\n\nNot modelled: the Chuansha Express (an OSM relation exists but no timetable or headway was found), Changzhou B21 and Zhoushan Peak Express 5 (suspended in CPTOND), services not yet open in Nov 2026 (Shanghai Lines 13 west and 22, Jiaxing-Fengnan suburban railway due 2027-28, Wuxi Line 4 phase 2, Suzhou Line 2/4/7 northern extensions), "
          "the Hangzhou-Haining express (6 pairs a day) and the Zhangjiang tram (closed June 2023).\n")
    write_feed(ROOT / "shanghai" / "gtfs" / "assumed_osm_rapid_transit", "OpenStreetMap + operator notices, see ASSUMED.md", "Shanghai metro area rapid transit missing from CPTOND (assumed)", routes, md)


if __name__ == "__main__":
    which = sys.argv[1:] or ["guadalajara", "concepcion"]
    if "shanghai" in which:
        build_shanghai_intercity_rail()
        build_shanghai_osm_rapid_transit()
    if "guadalajara" in which:
        build_guadalajara()
        build_guadalajara_osm_extra()
        build_guadalajara_treep()
        build_guadalajara_line4()
        build_guadalajara_line5()
    if "concepcion" in which:
        build_concepcion()
        build_concepcion_osm_extra()
        build_concepcion_santa_juana()
