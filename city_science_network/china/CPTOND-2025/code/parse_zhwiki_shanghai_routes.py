"""Parse zh.wikipedia "上海市公交线路列表 (数字线路)" into a per-route CSV.

Input : shanghai/gtfs/sources/zhwiki_shanghai_bus_routes_numbered.html
Output: shanghai/gtfs/sources/zhwiki_shanghai_numbered_routes_parsed.csv
        (route number, section heading, per-direction first/last times,
         headway quoted in the notes if any, raw row text)

Each table row reads "<number> <terminal A> <times> ⇆ <terminal B> <times>
<fare> <operator> <notes>"; times are "H:MM–H:MM" (several ranges for
peak-only lines). Stdlib only (no lxml in this environment).
"""

from __future__ import annotations

import csv
import os
import re
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "..", "..", "..", "shanghai", "gtfs", "sources")
SRC = os.path.join(SOURCES, "zhwiki_shanghai_bus_routes_numbered.html")
OUT = os.path.join(SOURCES, "zhwiki_shanghai_numbered_routes_parsed.csv")

TIME_RANGE = re.compile(r"(\d{1,2}:\d{2})\s*[–—-]\s*(\d{1,2}:\d{2})")
HEADWAY = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|~|至)?\s*(\d+(?:\.\d+)?)?\s*分钟?\s*/\s*班")


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.heading, self.in_heading, self.heading_buf = [], None, []
        self.rows, self.row, self.cell, self.in_cell = [], None, [], False
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script", "sup"):
            self.skip += 1
        if tag in ("h2", "h3", "h4"):
            self.in_heading, self.heading_buf = tag, []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.in_cell, self.cell = True, []
        elif tag == "br" and self.in_cell:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("style", "script", "sup"):
            self.skip = max(0, self.skip - 1)
        if tag == self.in_heading:
            text = re.sub(r"\[\s*编辑\s*\]", "", "".join(self.heading_buf)).strip()
            level = int(tag[1])
            self.heading = self.heading[: level - 2] + [text]
            self.in_heading = None
        elif tag in ("td", "th") and self.in_cell:
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.in_cell = False
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append((" / ".join(self.heading), self.row))
            self.row = None

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_heading:
            self.heading_buf.append(data)
        if self.in_cell:
            self.cell.append(data)


def main():
    p = Tables()
    p.feed(open(SRC, encoding="utf-8").read())
    out = []
    for section, cells in p.rows:
        cells = [c for c in cells if c]  # drop the empty colour-bar cell
        m = re.fullmatch(r"(\d{1,4})\s*(.*)", cells[0]) if cells else None
        if not m or "⇆" not in " ".join(cells) and not TIME_RANGE.search(" ".join(cells)):
            continue
        body = " ".join(cells[1:])
        if "⇆" in body:
            a, b = body.split("⇆", 1)
        else:
            a, b = body, ""
        ra = [f"{s}-{e}" for s, e in TIME_RANGE.findall(a)]
        rb = [f"{s}-{e}" for s, e in TIME_RANGE.findall(b)]
        hw = HEADWAY.search(body)
        headway = ""
        if hw:
            lo, hi = float(hw.group(1)), float(hw.group(2) or hw.group(1))
            headway = f"{(lo + hi) / 2:g}"
        out.append({"number": m.group(1), "label": m.group(2), "section": section, "times_a": ";".join(ra),
                    "times_b": ";".join(rb), "headway_min": headway, "raw": " | ".join(cells)})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
