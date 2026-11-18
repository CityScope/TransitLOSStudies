# -*- coding: utf-8 -*-
"""Source, native level and description (English + Spanish) of every column of the Spain map, for the "i" panels and the Metadata tab.

``describe(col)`` -> ``{"name", "source", "level", "description", "unit"}`` for English and the same for Spanish. The wording follows the
pycensus schemas (``countries/spain/schema/*_schema.json``) and loaders; every text is in ``i18n/column_docs.json``; party lists of the left/right blocs come from ``elections_blocs.json``.
``inject_column_docs(html, columns)`` rewrites ``window.__columnMeta`` of a finished map.html and adds ``window.__columnMetaEs``.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from pycensus.schema import schema_dir  # noqa: E402

BLOCS = schema_dir("spain") / "elections_blocs.json"

_DOCS = json.loads((Path(__file__).resolve().parents[1] / "i18n" / "column_docs.json").read_text(encoding="utf-8"))   # every text of this module

# sources, native levels: key -> (English, Spanish)
SRC = {k: tuple(v) for k, v in _DOCS["sources"].items()}
LV = {k: tuple(v) for k, v in _DOCS["levels"].items()}


def _t(en, es):
    return {"en": en, "es": es}


# (name_en, name_es, source, level, desc_en, desc_es, unit) for columns that are documented one by one. "name" mirrors the labels in i18n/columns.json.
ONE = {k: tuple(v) for k, v in _DOCS["columns"].items()}
_BUILT = {k: tuple(v) for k, v in _DOCS["built_bands"].items()}
_AREA_NAME = {k: tuple(v) for k, v in _DOCS["floor_area_bands"].items()}
_BLDG = {k: tuple(v) for k, v in _DOCS["building_types"].items()}
_TPL = _DOCS["templates"]
_EL = _DOCS["elections"]
_MI = _DOCS["mitma"]


def _fill(tpl, **kw):
    """A template row (list of strings / codes) with its {placeholders} filled."""
    return tuple(t.format(**kw) if isinstance(t, str) and "{" in t else t for t in tpl)


def _blocs():
    d = json.loads(BLOCS.read_text(encoding="utf-8"))
    out = {}
    for sc, v in d.items():
        if isinstance(v, dict) and "parties" in v:
            g = defaultdict(list)
            for k, p in v["parties"].items():
                g[p.get("bloc", "other")].append(k)
            out[sc.split("_")[0]] = {b: sorted(l) for b, l in g.items()}
    return out


_BL = None
_ELEC = {k: tuple(v) for k, v in _EL["scopes"].items()}
_ELEC_RE = re.compile(r"^(?:ine_)?elec_(national|municipal|state)(?:_(\d{4}))?_(.+)$")


def _elec(col, with_year=False):
    global _BL
    m = _ELEC_RE.match(col)
    if not m:
        return None
    _BL = _BL or _blocs()
    scope, year, what = m.groups()
    e_en, e_es = _ELEC[scope]
    bl = _BL.get(scope, {})
    L, R = ", ".join(bl.get("left", [])), ", ".join(bl.get("right", []))
    O = ", ".join(bl.get("other", [])) or "-"
    src = "elec"
    tpl = _EL["base"].get(what)
    base = _fill(tpl, e_en=e_en.lower(), e_es=e_es.lower(), L=L, R=R, O=O) if tpl else None
    if base is None:
        pm = re.match(r"party_(.+)_(votes|share)$", what)
        if not pm:
            return None
        slug, kind = pm.groups()
        from .i18n import party_name
        pn = party_name(slug)
        bloc = next((b for b, l in bl.items() if any(re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_") == slug for x in l)), None)
        bloc = bloc if bloc in ("left", "right") else None
        bk = bloc or "other"
        base = _fill(_EL["party_votes" if kind == "votes" else "party_share"], pn=pn, bloc_en=_EL["blocs_en"][bk], bloc_es=_EL["blocs_es"][bk])
    en_n, es_n, en_d, es_d = base
    year_txt = f" {year}" if (year and with_year) else ""
    return {"en": (f"{e_en}{year_txt}: {en_n}", en_d), "es": (f"{e_es}{year_txt}: {es_n}", es_d), "src": src, "level": "section",
            "unit": "%" if what.endswith(("_share", "turnout_rate")) else "votes"}


def _mitma(col):
    """Row for an ``ine_mitma_*`` column (same tuple layout as ONE), or None."""
    from .i18n import COLUMNS
    if not col.startswith("ine_mitma_") or col not in COLUMNS:
        return None
    ne, ns = COLUMNS[col]
    two, dos = _MI["weekdays"]
    home, casa = _MI["home"]
    if "regular_flows" in col:
        pe, ps = _MI["regular_flows"]
        if "up_to_3" in col:
            k = col.rsplit("_", 1)[1]
            if k == "est":
                de, ds = _MI["up_to_3_est"][0] + pe, _MI["up_to_3_est"][1] + ps
            else:
                w, bound, ww, bound_es = _MI["bound_words"]["min" if k == "min" else "max"]
                de = _MI["up_to_3_bound"][0].format(w=w, bound=bound) + pe
                ds = _MI["up_to_3_bound"][1].format(w=ww, bound=bound_es) + ps
            return (ne, ns, "mitma", "mitma_district", de, ds, "%")
        if "_share_" in col:
            return (ne, ns, "mitma", "mitma_district", _MI["regular_share"][0] + pe, _MI["regular_share"][1] + ps, "%")
        return (ne, ns, "mitma", "mitma_district", _MI["regular_count"][0] + pe, _MI["regular_count"][1] + ps, "trip-pairs")
    if col.endswith(("trips_per_person", "distance_km_per_person")):
        km = "distance" in col
        lead = _MI["km_per_person" if km else "trips_per_person"]
        return (ne, ns, "mitma", "mitma_district", lead[0] + two + home + _MI["per_person_tail"][0], lead[1] + dos + casa + _MI["per_person_tail"][1],
                "km" if km else "trips")
    if "long_distance" in col:
        w_en, w_es = _MI["long_distance_what"]["km" if "km_share" in col else "trips"]
        return (ne, ns, "mitma", "mitma_district", _MI["long_distance"][0].format(what_en=w_en) + two + home,
                _MI["long_distance"][1].format(what_es=w_es) + dos + casa, "%")
    return (ne, ns, "mitma", "mitma_district", _MI["distance_band"][0] + two + home, _MI["distance_band"][1] + dos + casa, "%")


def _strip_year(col):
    c = re.sub(r"_(?:19|20)\d\d(?:M\d\d)?(?=_|$)", "", col)
    return c


def describe(col, with_year=False):
    """``{"en": {...}, "es": {...}}`` with name/source/level/description/unit, or None for a column this module does not know."""
    e = _elec(col, with_year)
    if e:
        return {l: {"name": e[l][0], "source": SRC[e["src"]][0 if l == "en" else 1], "level": LV[e["level"]][0 if l == "en" else 1], "description": e[l][1], "unit": e["unit"]} for l in ("en", "es")}
    row = ONE.get(col) or ONE.get(_strip_year(col)) or _mitma(col)
    if row is None:
        m = re.match(r"^ine_valor_tasado_(total|new|old)_eur_m2_\d{4}$", col)
        if m:
            row = ONE[f"ine_valor_tasado_{m.group(1)}_eur_m2"]
    if row is None:
        m = re.match(r"^ine_built_(.+)_share$", col)
        if m:
            k = m.group(1)
            en, es = _BUILT.get(k) or ((k.replace("_a_", "-"),) * 2)
            if k not in _BUILT:
                a, b = k.split("_a_") if "_a_" in k else (k, k)
                en = es = f"{a}-{b}" if "_a_" in k else k
            row = _fill(_TPL["built"], en=en, es=es)
        m = re.match(r"^ine_floor_area_(.+?)(?:_m2)?_share$", col)
        if m and row is None:
            k = m.group(1)
            if k == "No_Consta":
                row = tuple(_TPL["floor_area_not_stated"])
            else:
                en, es = _AREA_NAME.get(k) or ((k.replace("_", "-"),) * 2)
                row = _fill(_TPL["floor_area"], en=en, es=es)
        m = re.match(r"^ine_building_(.+)_share$", col)
        if m and row is None and m.group(1) in _BLDG:
            en, es = _BLDG[m.group(1)]
            row = _fill(_TPL["building"], en=en, es=es)
        m = re.match(r"^dwellings_built_(before_1945|1945_1970|1970_2000|after_2000)_share$", col)
        if m and row is None:
            k = m.group(1)
            en, es = _DOCS["dwellings_built_groups"][k]
            row = _fill(_TPL["dwellings_built_group"], en=en, es=es)
    if row is None:
        return None
    ne, ns, src, lv, de, ds, unit = row
    from .i18n import COLUMNS, simple
    if col in COLUMNS:
        ne, ns = simple(COLUMNS[col][0]), simple(COLUMNS[col][1])
    ne, ns = simple(ne), simple(ns)
    return {"en": {"name": ne, "source": SRC[src][0], "level": LV[lv][0], "description": de, "unit": unit},
            "es": {"name": ns, "source": SRC[src][1], "level": LV[lv][1], "description": ds, "unit": unit}}


def inject_column_docs(html_path, columns=None):
    """Replace ``window.__columnMeta`` by complete English entries and add ``window.__columnMetaEs``; returns the columns left undocumented."""
    p = Path(html_path)
    s = p.read_text(encoding="utf-8")
    m = re.search(r"window\.__columnMeta\s*=\s*", s)
    if not m:
        return []
    meta, end = json.JSONDecoder().raw_decode(s[m.end():])
    from .i18n import _multi_dated
    multi = _multi_dated(list(meta))
    es, missing = {}, []
    for col in list(meta):
        d = describe(col, col in multi)
        if d is None:
            missing.append(col)
            continue
        meta[col] = {**meta[col], **d["en"]}
        es[col] = d["es"]
    s = s[:m.end()] + json.dumps(meta, ensure_ascii=False).replace("</", "<\\/") + s[m.end() + end:]
    s = s.replace("window.__columnMeta =", "window.__columnMetaEs = " + json.dumps(es, ensure_ascii=False).replace("</", "<\\/") + ";\nwindow.__columnMeta =", 1) \
        if "window.__columnMetaEs" not in s else s
    s = _rewrite_metadata_table(s, meta, multi)
    p.write_text(s, encoding="utf-8")
    return missing


def _rewrite_metadata_table(s: str, meta: dict, multi) -> str:
    """Metadata tab: the label, native level and description of every row come from `describe` (English in the page, Spanish through the
    translator dictionary `i18n.EXTRA_ES`), so the table says the same as the column's "i" panel."""
    import html as _html

    from . import i18n

    i = s.find('id="metadataTabBody"')
    if i < 0:
        return s
    j = s.index("</tbody>", i)
    body = s[i:j]
    cell_re = re.compile(r"<td([^>]*?)>(.*?)</td>", re.S)
    extra = getattr(i18n, "EXTRA_ES", None)
    if extra is None:
        extra = i18n.EXTRA_ES = {}
    norm = lambda t: " ".join(str(t).split())   # noqa: E731

    def fix_row(m):
        row = m.group(0)
        cells = list(cell_re.finditer(row))
        if len(cells) < 14:
            return row
        key = _html.unescape(cells[1].group(2)).strip()
        d = describe(key, key in multi) if key in meta else None
        if d is None:
            return row
        new = {0: (d["en"]["name"], d["es"]["name"]), 10: (d["en"]["level"], d["es"]["level"]), 13: (d["en"]["description"], d["es"]["description"])}
        out, last = [], 0
        for k, c in enumerate(cells):
            out.append(row[last:c.start()])
            if k in new:
                en, es = new[k]
                extra[norm(en)] = norm(es)
                attrs = re.sub(r"title='[^']*'", "title='" + _html.escape(en, quote=True).replace("'", "&#39;") + "'", c.group(1))
                out.append("<td%s>%s</td>" % (attrs, _html.escape(en, quote=False)))
            else:
                out.append(c.group(0))
            last = c.end()
        out.append(row[last:])
        return "".join(out)

    body = re.sub(r"<tr[^>]*>.*?</tr>", fix_row, body, flags=re.S)
    return s[:i] + body + s[j:]


if __name__ == "__main__":
    import sys
    sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
    print(inject_column_docs(sys.argv[1]))
