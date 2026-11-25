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


_ESM = """
import maplibregl from "https://esm.sh/maplibre-gl@5.13.0";
import MapboxDraw from "https://esm.sh/@mapbox/mapbox-gl-draw@1.4.3";

function css(el, href) { const l = document.createElement("link"); l.rel = "stylesheet"; l.href = href; el.appendChild(l); }

function render({ model, el }) {
  css(el, "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.css");
  const draw = model.get("draw");
  if (draw) css(el, "https://unpkg.com/@mapbox/mapbox-gl-draw@1.4.3/dist/mapbox-gl-draw.css");
  const box = document.createElement("div");
  box.style.cssText = "width:100%;height:" + model.get("height") + "px";
  el.appendChild(box);
  const b = model.get("bounds");
  const map = new maplibregl.Map({ container: box, style: "https://tiles.openfreemap.org/styles/positron",
                                   bounds: [[b[0], b[1]], [b[2], b[3]]], fitBoundsOptions: { padding: 24 } });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }));
  const outline = model.get("outline");
  if (outline) map.on("load", () => {
    map.addSource("area", { type: "geojson", data: outline });
    map.addLayer({ id: "area-fill", type: "fill", source: "area", paint: { "fill-color": "#d33", "fill-opacity": 0.1 } });
    map.addLayer({ id: "area-line", type: "line", source: "area", paint: { "line-color": "#d33", "line-width": 2 } });
  });
  if (draw) {
    const c = MapboxDraw.constants.classes;   // mapbox-gl-draw with maplibre: its control classes
    c.CONTROL_BASE = "maplibregl-ctrl"; c.CONTROL_PREFIX = "maplibregl-ctrl-"; c.CONTROL_GROUP = "maplibregl-ctrl-group";
    const tool = new MapboxDraw({ displayControlsDefault: false, controls: { polygon: true, trash: true } });
    map.addControl(tool, "top-left");
    const sync = () => { model.set("drawn", tool.getAll().features); model.save_changes(); };
    ["draw.create", "draw.update", "draw.delete"].forEach((e) => map.on(e, sync));
  }
}
export default { render };
"""


def _widget():
    import anywidget
    import traitlets

    class AreaMap(anywidget.AnyWidget):
        """A MapLibre map with the OpenMapTiles background (OpenFreeMap), an area outline, and (`draw`) a polygon drawing tool."""

        _esm = _ESM
        bounds = traitlets.List([-10, -10, 10, 10]).tag(sync=True)
        outline = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
        draw = traitlets.Bool(False).tag(sync=True)
        height = traitlets.Int(420).tag(sync=True)
        drawn = traitlets.List([]).tag(sync=True)   # the shapes drawn (GeoJSON features)

    return AreaMap


_STATS_ONLY = """
<style id="ptlos-stats-only">
html, body { overflow: auto !important; }
html body #statsButton, html body #top-center-bar, html body #layer-switcher, html body #routeEditToggleBtn, html body #route-toolbar,
html body #downloadButton, html body #downloadPanel, html body .maplibregl-control-container { display: none !important; }
html body #statsPanel { display: flex !important; flex-direction: column !important; position: fixed !important; left: 0 !important; top: 0 !important;
  right: 0 !important; bottom: 0 !important; width: auto !important; height: auto !important; max-width: none !important; max-height: none !important;
  transform: none !important; margin: 0 !important; border: 0 !important; border-radius: 0 !important; box-shadow: none !important;
  resize: none !important; z-index: 20000 !important; }
</style>
<script>
(function () {   // the map page with its statistics panel opened over the whole frame, once the statistics data has loaded
  var n = 0, t = setInterval(function () {
    var b = document.getElementById('statsButton'), p = document.getElementById('statsPanel');
    if (p && b && window.__statsData) {
      clearInterval(t);
      if (getComputedStyle(p).display === 'none') b.click();
      setTimeout(function () { var a = p.querySelector('.statsTabBtn.active'); if (a) a.click(); }, 600);   // draws the tab now that the data is there
    } else if (++n > 300) clearInterval(t);
  }, 300);
})();
</script>
"""


def stats_page(results) -> Path:
    """`stats.html` next to the run's `map.html`: the same page with the statistics panel (distribution, regression, ANOVA, discretization,
    metadata) opened over the whole frame, to show in a cell of its own with more room than beside the map."""
    area_dir = Path(results).parent if str(results).endswith(".zip") else Path(results)
    page = (area_dir / "map.html").read_text(encoding="utf-8")
    out = area_dir / "stats.html"
    out.write_text(page.replace("</body>", _STATS_ONLY + "</body>", 1) if "</body>" in page else page + _STATS_ONLY, encoding="utf-8")
    return out


def show_stats(results, height: int = 900):
    """Show the statistics panel of a run in its own frame (see `stats_page`)."""
    stats_page(results)
    area_dir = Path(results).parent if str(results).endswith(".zip") else Path(results)
    port = _serve(area_dir)
    try:
        from google.colab import output
    except ImportError:
        from IPython.display import IFrame

        return IFrame(f"http://localhost:{port}/stats.html", width="100%", height=height)
    output.serve_kernel_port_as_iframe(port, path="/stats.html", height=height)


def draw_area(area: Optional[dict] = None):
    """A map with a polygon drawing tool, opened on `area`. The shapes drawn are in its `.drawn` (GeoJSON features)."""
    return _widget()(draw=True, height=520, bounds=area["bounds"] if area else [-180, -60, 180, 75])


def show_area(area: dict):
    """The area on the OpenMapTiles map (a simplified outline: big geometries stay light), with its name and size."""
    import ipywidgets as widgets

    return widgets.VBox([widgets.HTML(f"<b>{area['name']}</b>: {area['km2']:,.0f} km²"),
                         _widget()(bounds=area["bounds"], outline=area["outline"])])


def select_sources(catalog: list):
    """The census data sources, to tick BEFORE the run: only the ticked ones are downloaded (locked ones are always needed). One line each: the
    source, what it holds, and what it costs. Its `.value` is the list of ticked keys, for `run(census_sources=...)`."""
    import html

    import ipywidgets as widgets

    L, grey = widgets.Layout, "color:#777;font-size:11px"
    rows, boxes = [], []
    for r in catalog:
        cb = widgets.Checkbox(value=bool(r["default"]), disabled=bool(r["required"]), description=r["title"], indent=False, layout=L(width="300px", flex="0 0 auto", margin="0"))
        text = r["description"] + (f" <i>{r['note']}.</i>" if r["note"] else "") + (" <i>Always used.</i>" if r["required"] else "")
        rows.append(widgets.HBox([cb, widgets.HTML(f"<div style='{grey};line-height:1.25'>{text}</div>", layout=L(margin="0"))], layout=L(align_items="center", margin="0 0 2px 0")))
        boxes.append((cb, r["key"]))
    count = widgets.HTML()

    def update(*_):
        count.value = f"<span style='{grey}'>{sum(b.value for b, _ in boxes)} of {len(boxes)} sources: the unticked ones are not downloaded</span>"

    for b, _ in boxes:
        b.observe(update, "value")
    update()

    class SourceChoice(widgets.VBox):
        @property
        def value(self) -> list:
            return [k for b, k in boxes if b.value]

    return SourceChoice([*rows, count])


def select_candidates(candidates: list):
    """The feeds the sources list for the area, from their metadata, before anything is downloaded: tick the ones to download. Those that
    probably serve the area start ticked. Its `.value` is the list of keys to give to `list_feeds(selected=...)` (None without candidates)."""
    import html

    import ipywidgets as widgets

    L, small = widgets.Layout, "color:#666;font-size:90%"
    if not candidates:
        return type("Own", (), {"value": None})()   # your own feeds: nothing to choose before downloading
    rows, boxes = [], []
    for e in candidates:
        cb = widgets.Checkbox(value=bool(e["likely"]), description="", indent=False, layout=L(width="26px", flex="0 0 auto"))
        facts = " · ".join(x for x in (e["source"], e["location"], e["modes"], f"updated {e['updated']}" if e["updated"] else "") if x)
        info = widgets.HTML(f"<b>{html.escape(e['provider'])}</b><br><span style='{small}'>{html.escape(facts)}</span><br>"
                            f"<span style='{small};font-style:italic'>{html.escape(e['why'])}</span>", layout=L(flex="1 1 auto"))
        rows.append(widgets.HBox([cb, info], layout=L(margin="2px 0 8px 0", align_items="flex-start", flex="0 0 auto")))
        boxes.append((cb, e))
    tick, untick = widgets.Button(description="Tick all", layout=L(width="90px")), widgets.Button(description="Untick all", layout=L(width="90px"))
    count = widgets.HTML()

    def update(*_):
        count.value = f"<span style='{small}'>{sum(b.value for b, _ in boxes)} of {len(boxes)} feeds ticked</span>"

    def set_all(value):
        for b, _ in boxes:
            b.value = value

    for b, _ in boxes:
        b.observe(update, "value")
    tick.on_click(lambda _: set_all(True))
    untick.on_click(lambda _: set_all(False))
    update()

    class CandidateChoice(widgets.VBox):
        @property
        def value(self) -> list:
            keys = []
            for b, e in boxes:
                if b.value:   # the most local source's rows of the feed (the others list the same feed: not downloaded twice)
                    keys += [k for k in e["keys"] if k.split(":")[0] == e["key"].split(":")[0]]
            return keys

    return CandidateChoice([widgets.HBox([tick, untick, count], layout=L(align_items="center", margin="0 0 8px 0")),
                            widgets.VBox(rows, layout=L(max_height="460px", overflow_y="auto", border="1px solid #ddd", padding="6px"))])


def select_feeds(feeds: list):
    """A checklist of the feeds (all ticked). Its `.value` is the list of the ticked feeds' names."""
    import ipywidgets as widgets

    rows = []
    for f in feeds:
        span = f" · calendar {f['first_date']} to {f['last_date']}" if f["first_date"] else ""
        info = f"{f['source']} · {f['stops']:,} stops in the area · {f['trips']:,} trips on a typical day{span}"
        box = widgets.Checkbox(value=True, description=f["agency"] or f["name"], indent=False, layout=widgets.Layout(width="320px"))
        box.feed = f["name"]
        rows.append(widgets.HBox([box, widgets.HTML(f"<span style='color:#666;font-size:90%'>{info}</span>")]))

    class FeedChoice(widgets.VBox):
        @property
        def value(self) -> list:
            return [r.children[0].feed for r in self.children if r.children[0].value]

    return FeedChoice(rows)


def pick_date(info: dict):
    """The service on every date of the chosen feeds, as a plot (bars: stop calls in the area, i.e. stops x trips). Weekdays, Saturdays,
    Sundays and public holidays have their own colour; the day type chosen in the settings is drawn strongest. The date to use is in a
    date box next to it, first the suggested one; its `.value` is the date ("YYYY-MM-DD") and the red line follows it."""
    import datetime

    import ipywidgets as widgets
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from IPython.display import display
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = info["dates"]
    dates = [datetime.date.fromisoformat(r["date"]) for r in rows]
    events = [r["events"] for r in rows]
    holiday = [r.get("holiday", "") for r in rows]
    colours = {"weekday": "#2a6fb0", "saturday": "#e69f00", "sunday": "#9b59b6"}
    kind = ["holiday" if h else ("saturday" if d.weekday() == 5 else "sunday" if d.weekday() == 6 else "weekday") for d, h in zip(dates, holiday)]
    paint = [colours.get(k, "#c0392b") for k in kind]
    strong = [k == info["day"] for k in kind]
    suggested = datetime.date.fromisoformat(info["suggested"])
    box = widgets.DatePicker(description="Service date", value=suggested, style={"description_width": "initial"})
    out = widgets.Output()

    def draw(_=None):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(11, 3.8))
            for d, e, c, k, s in zip(dates, events, paint, kind, strong):
                ax.bar(d, e, width=1.0, color=c, alpha=1.0 if s else 0.55, hatch="////" if k == "holiday" else None,
                       edgecolor="white" if k != "holiday" else "#7b241c", linewidth=0.8 if k != "holiday" else 0.4)
            top = max(events) if events else 1
            real = [d for d, e in zip(dates, events) if e >= 0.05 * top]   # where there is real service (tiny feeds can run for years)
            lo, hi = (min(real) - datetime.timedelta(days=7), max(real) + datetime.timedelta(days=7)) if real else (dates[0], dates[-1])
            ax.set_xlim(lo, hi)
            for d, e, h in zip(dates, events, holiday):
                if h and lo + datetime.timedelta(days=2) <= d <= hi:
                    ax.text(d, e + top * 0.02, h, rotation=90, ha="center", va="bottom", fontsize=7, color="#7b241c")
            ax.set_ylim(0, top * 1.4)
            ticks = mdates.AutoDateLocator(minticks=4, maxticks=9)
            ax.xaxis.set_major_locator(ticks)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ticks))
            ax.axvline(suggested, color="#444", ls="--", lw=1)
            if box.value:
                ax.axvline(box.value, color="#d33", lw=2)
            ax.set_ylabel("stops × trips in the area")
            ax.set_title("Service on every date", loc="left", fontsize=10)
            handles = [Patch(color=colours["weekday"], label="weekday"), Patch(color=colours["saturday"], label="Saturday"),
                       Patch(color=colours["sunday"], label="Sunday"), Patch(facecolor="white", edgecolor="#7b241c", hatch="////", label="public holiday"),
                       Line2D([], [], color="#444", ls="--", label=f"suggested {suggested}")]
            if box.value:
                handles.append(Line2D([], [], color="#d33", lw=2, label=f"chosen {box.value}"))
            ax.legend(handles=handles, frameon=False, fontsize=8, ncol=6, loc="upper left", bbox_to_anchor=(0, 1.0))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    box.observe(draw, "value")
    draw()

    class DateChoice(widgets.VBox):
        @property
        def value(self):
            return str(box.value) if box.value else None

    return DateChoice([box, out])


def select_columns(groups: dict):
    """A compact chooser of the census columns. The topics are a grid of tick boxes (all ticked); the arrow next to a topic opens its columns
    below, one per line (name, unit, census level, source; the description on hover, or inline with *descriptions*) to tick single ones. The
    search box lists the matching columns of every topic. Its `.value` is the list of ticked columns (for `export_results`)."""
    import html

    import ipywidgets as widgets

    L, grey = widgets.Layout, "color:#777"
    esc = lambda t: html.escape(str(t), quote=True)   # noqa: E731
    entries, state = [], {"busy": False, "open": None}
    topic_box, topic_title = {}, {}
    detail = widgets.VBox(layout=L(max_height="340px", overflow_y="auto", border="1px solid #ddd", padding="3px 6px", margin="4px 0 0 0", display="none"))
    show_desc = widgets.Checkbox(value=False, description="descriptions", indent=False, layout=L(width="auto", margin="0 0 0 auto"))

    for name, g in groups.items():
        for c in g["columns"]:
            cb = widgets.Checkbox(value=True, description="", indent=False, layout=L(width="20px", flex="0 0 auto", margin="0"))
            meta = " · ".join(x for x in (c["unit"], c["level"], c["source"]) if x)
            tip = f"{c['column']}: {c['description']}" + (f" (in {c['in']})" if c["in"] else "")
            shown = c["name"][:-len(f" ({c['source']})")] if c["source"] and c["name"].endswith(f" ({c['source']})") else c["name"]   # the source is shown after it
            line = widgets.HTML(f"<div style='font-size:12px;line-height:1.3' title='{esc(tip)}'>{html.escape(shown)} <span style='{grey}'>{html.escape(meta)}</span></div>",
                                layout=L(margin="0"))
            desc = widgets.HTML(f"<div style='font-size:11px;{grey};margin:0 0 3px 20px'><code>{html.escape(c['column'])}</code> {html.escape(c['description'])}</div>",
                                layout=L(display="none", margin="0"))
            row = widgets.VBox([widgets.HBox([cb, line], layout=L(align_items="center", margin="0")), desc], layout=L(margin="0", flex="0 0 auto"))
            entries.append({"column": c["column"], "cb": cb, "row": row, "desc": desc, "topic": name,
                            "text": f"{c['name']} {c['column']} {c['description']} {c['source']} {name}".lower()})

    def refresh(topic):
        mine = [e["cb"] for e in entries if e["topic"] == topic]
        on = sum(b.value for b in mine)
        state["busy"] = True
        topic_box[topic].value = on == len(mine)
        state["busy"] = False
        topic_box[topic].description = f"{topic_title[topic]} ({on}/{len(mine)})" if on != len(mine) else f"{topic_title[topic]} ({len(mine)})"
        count.value = f"<span style='font-size:12px;{grey}'>{sum(e['cb'].value for e in entries)} of {len(entries)} columns</span>"

    def show_topic(topic):
        state["open"] = topic
        head = widgets.HBox([widgets.HTML(f"<span style='font-size:12px'><b>{html.escape(topic)}</b> <span style='{grey}'>{html.escape(groups[topic]['description'])}</span></span>"),
                             show_desc], layout=L(align_items="center", margin="0 0 3px 0", flex="0 0 auto"))
        detail.children = [head, *[e["row"] for e in entries if e["topic"] == topic]]
        detail.layout.display = ""

    def whole(change, topic):
        if not state["busy"]:
            for e in entries:
                if e["topic"] == topic:
                    e["cb"].value = change["new"]

    cells = []
    for name in groups:
        topic_title[name] = name
        box = widgets.Checkbox(value=True, description=f"{name} ({len(groups[name]['columns'])})", indent=False, layout=L(width="auto", flex="0 1 auto", margin="0"))
        arrow = widgets.Button(description="▸", tooltip=groups[name]["description"], layout=L(width="26px", height="22px", padding="0", margin="0"))
        topic_box[name] = box
        box.observe(lambda ch, t=name: whole(ch, t), "value")
        arrow.on_click(lambda _, t=name: show_topic(t))
        cells.append(widgets.HBox([box, arrow], layout=L(align_items="center", justify_content="flex-start", margin="0")))
    grid = widgets.GridBox(cells, layout=L(grid_template_columns="repeat(3, minmax(0, 1fr))", grid_gap="1px 14px", width="100%"))

    search = widgets.Text(placeholder="Search columns", layout=L(width="220px"))
    all_btn, none_btn = widgets.Button(description="All", layout=L(width="48px")), widgets.Button(description="None", layout=L(width="52px"))
    count = widgets.HTML()

    def set_all(value):
        for e in entries:
            e["cb"].value = value

    def searched(change):
        q = change["new"].strip().lower()
        if q:
            hits = [e for e in entries if q in e["text"]]
            detail.children = [widgets.HTML(f"<span style='font-size:12px;{grey}'>{len(hits)} matching columns" + (" (first 150 shown)" if len(hits) > 150 else "") + "</span>",
                                            layout=L(flex="0 0 auto")),
                               *[e["row"] for e in hits[:150]]]
            detail.layout.display = ""
        elif state["open"]:
            show_topic(state["open"])
        else:
            detail.layout.display = "none"

    def toggle_desc(change):
        for e in entries:
            e["desc"].layout.display = "" if change["new"] else "none"

    all_btn.on_click(lambda _: set_all(True))
    none_btn.on_click(lambda _: set_all(False))
    search.observe(searched, "value")
    show_desc.observe(toggle_desc, "value")
    for e in entries:
        e["cb"].observe(lambda ch, t=e["topic"]: refresh(t), "value")
    for name in groups:
        refresh(name)

    class ColumnChoice(widgets.VBox):
        @property
        def value(self) -> list:
            return [e["column"] for e in entries if e["cb"].value]

    top = widgets.HBox([search, all_btn, none_btn, count], layout=L(align_items="center", margin="0 0 4px 0"))
    return ColumnChoice([top, grid, detail])


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
