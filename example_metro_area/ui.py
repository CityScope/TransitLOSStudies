"""Notebook display helpers: maps, drawing, uploads, downloads. Light on purpose (no geopandas, no study packages): they run in the
notebook's own kernel, while the study itself runs in a separate process (`bootstrap.py`, `worker.py`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def show_map(results, height: int = 760):
    """Show the interactive map of a run (`map.html`, with its distribution, regression, ANOVA, discretization and metadata tabs) in
    Jupyter or Colab. The map files are served from this machine by a small local server.

    Args:
        results: The results zip returned by `run`, or the area folder.
        height: Height of the map in pixels.
    """
    area_dir = Path(results).parent if str(results).endswith(".zip") else Path(results)
    if not (area_dir / "map.html").is_file():
        raise FileNotFoundError(f"No map in {area_dir}: run with build_map=True.")
    port = _serve(area_dir)
    try:
        from google.colab import output
    except ImportError:
        from IPython.display import IFrame

        return IFrame(f"http://localhost:{port}/map.html", width="100%", height=height)
    output.serve_kernel_port_as_iframe(port, path="/map.html", height=height)


_SERVERS: dict = {}


def _serve(folder: Path) -> int:
    """Serve `folder` over HTTP on a free local port (in a background thread) and return the port. Supports byte ranges, which the
    map's tile files (PMTiles) need."""
    import functools
    import http.server
    import threading

    folder = Path(folder).resolve()
    if folder in _SERVERS:
        return _SERVERS[folder]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):   # quiet
            pass

        def send_head(self):
            rng = self.headers.get("Range")
            path = Path(self.translate_path(self.path))
            if not rng or not path.is_file():
                return super().send_head()
            size = path.stat().st_size
            first, _, last = rng.split("=", 1)[1].split(",")[0].strip().partition("-")
            if first:            # bytes=first-last or bytes=first-
                start, end = int(first), min(int(last), size - 1) if last else size - 1
            else:                # bytes=-n: the last n bytes
                start, end = max(0, size - int(last)), size - 1
            if start >= size:
                self.send_error(416)
                return None
            f = open(path, "rb")
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._remaining = end - start + 1
            return f

        def copyfile(self, source, outputfile):
            left = getattr(self, "_remaining", None)
            if left is None:
                return super().copyfile(source, outputfile)
            while left > 0:
                chunk = source.read(min(1 << 16, left))
                if not chunk:
                    break
                outputfile.write(chunk)
                left -= len(chunk)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _SERVERS[folder] = server.server_address[1]
    return _SERVERS[folder]


# ---------------------------------------------------------------- notebook helpers


def draw_area(area: Optional[dict] = None):
    """A map with drawing tools (polygon, rectangle), opened on `area`. Returns (map, drawn): `drawn` collects the shapes drawn."""
    from ipyleaflet import DrawControl, Map

    minx, miny, maxx, maxy = area["bounds"] if area else (-10, -10, 10, 10)
    drawn: list = []
    m = Map(center=((miny + maxy) / 2, (minx + maxx) / 2), zoom=9 if area else 2, scroll_wheel_zoom=True)
    ctl = DrawControl(polygon={"shapeOptions": {"color": "#d33"}}, rectangle={"shapeOptions": {"color": "#d33"}},
                      polyline={}, circlemarker={}, marker={}, circle={})
    ctl.on_draw(lambda control, action, geo_json: drawn.append(geo_json) if action == "created" else None)
    m.add(ctl)
    return m, drawn


def show_area(area: dict):
    """The area on a light map (ipyleaflet, a simplified outline: folium chokes on big geometries), with its name and size."""
    import ipywidgets as widgets
    from ipyleaflet import GeoJSON, Map

    minx, miny, maxx, maxy = area["bounds"]
    m = Map(center=((miny + maxy) / 2, (minx + maxx) / 2), scroll_wheel_zoom=True, layout=widgets.Layout(height="420px"))
    m.add(GeoJSON(data=area["outline"], style={"color": "#d33", "fillOpacity": 0.1, "weight": 2}))
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return widgets.VBox([widgets.HTML(f"<b>{area['name']}</b>: {area['km2']:,.0f} km²"), m])


def upload_file() -> str:
    """Pick one file from your computer (Colab); returns its name."""
    from google.colab import files

    return next(iter(files.upload()))


def upload_feeds() -> list:
    """Pick GTFS zip files from your computer (Colab); returns their names, for `FEEDS`."""
    from google.colab import files

    return list(files.upload())


def download(path) -> None:
    """Download a file in Colab; elsewhere print where it is."""
    try:
        from google.colab import files
    except ImportError:
        print(path)
    else:
        files.download(str(path))
