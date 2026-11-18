# -*- coding: utf-8 -*-
"""ES/EN strings of the Spain map (consumed by ``i18n_inject`` and ``column_docs``).

Every translation lives in the ``i18n/`` folder of this study (plain JSON, one file per kind); this module only loads
and combines them:
``UI_ES`` (ui_es.json, scoring_axes_es.json): the map's own interface texts, original English text exactly as printed -> Spanish.
``PATTERNS_ES`` (patterns_es.json): regex -> replacement for texts with numbers ("Show all 65 census variables").
``COLUMNS`` (columns.json): every census column key -> (English label, Spanish label). The map prints a humanised form of the key
("Serpavi rent m 2 median vc 2024"); ``humanize`` reproduces it and builds the lookup in both languages (``COLUMN_LABELS_EN`` rewrites the clumsy
English, ``COLUMN_LABELS_ES`` gives Spanish; the "(%)" unit suffix the map appends is handled). Election columns (elections.json) are labelled from
their key, so any party works. To add a column: add a row to i18n/columns.json and re-run ``tests/test_language_switch.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parents[1] / "i18n"


def _load(name: str):
    return json.loads((I18N_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------- interface
UI_ES = dict(_load("ui_es.json")["strings"])
PATTERNS_ES = _load("patterns_es.json")["patterns"]

# --------------------------------------------------------------------------------------------- columns
def humanize(key: str) -> str:
    """The label the map prints for a column key (verified against a built map by tests/test_language_switch.py)."""
    k = key[4:] if key.startswith("ine_") else key
    k = re.sub(r"([a-z])([A-Z])", r"\1 \2", k).replace("_", " ")
    k = re.sub(r"([A-Za-z])(\d)", r"\1 \2", k)
    k = re.sub(r"(\d)([A-Za-z])", r"\1 \2", k)
    k = re.sub(r"\s+", " ", k).strip().lower()
    return k[:1].upper() + k[1:]


def _builder_bases(key: str) -> list:
    """The labels the shared builder (``transitlos.map.build``) prints for a column: its display name (JS lists) and unit-suffixed field label."""
    out = [humanize(key)]
    try:
        from transitlos.map import build as b

        out += [b.display_field_name(key), b.field_label(key)]
    except Exception:  # noqa: BLE001 -- builder not importable (docs / tests): fall back to the local humaniser
        pass
    return list(dict.fromkeys(out))


def _unit(key: str) -> str:
    return " (%)" if re.search(r"(share|pct|ratio)(_|\d|$)", key.lower()) else ""


# key -> (English label, Spanish label); the "(%)" suffix is added automatically for share / pct / ratio columns.
COLUMNS: dict = {k: (e, s) for k, e, s in _load("columns.json")["columns"]}


#: Columns whose English label is ours rather than the builder's: municipality / district natives (the popup footnote keys on the "(municipality)" /
#: "(district)" note), Catastro (not in the builder's vocabulary) and the labels the user asked for explicitly. Everything else keeps the builder's English.
_OWN_ENGLISH = re.compile(r"^(?:(?:rent|price)_per_(?:m2|home)_mean|home_size_m2_mean|dwellings_built_\w+_share|elec_(?:national|state|municipal)_(?:left|right|other)_share|elec_(?:national|state|municipal)_(?:turnout_rate|total_votes))$|^ine_(tourist_|netmig_|valor_tasado|emal_|tenure_|built_|floor_area_|building_|catastro_|serpavi_n_vc_2024$)")


def _keep_english(key: str) -> bool:
    return bool(_OWN_ENGLISH.match(key)) or key == "level_of_service"


_DATE_RE = re.compile(r"[,\s]*\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|ene|abr|ago|dic)[a-z]*\.?\s+)?(?<![-–])20(?:0[5-9]|[12]\d|3\d)\b(?![-–]\d)", re.I)


def nodate(label: str) -> str:
    """Column labels carry no reference dates (the latest data only); "Median rent per m2, flats 2024 (SERPAVI)" -> "Median rent per m2, flats (SERPAVI)"."""
    return re.sub(r"\s+", " ", _DATE_RE.sub("", label)).replace(" ,", ",").strip()


_PAREN_RE = re.compile(r"\s*\((?!%\))[^()]*\)")


def simple(label: str) -> str:
    """Column labels stay short: no source, level, unit or date in brackets ("Mean useful area (Eustat)" -> "Mean useful area"); those live in the "i" panel."""
    return re.sub(r"\s+", " ", _PAREN_RE.sub("", nodate(label))).strip()


def _labels():
    en, es = {}, {}
    for key, (e, s) in COLUMNS.items():
        e0 = e
        e, s = simple(e), simple(s)
        for h in _builder_bases(key):
            for suffix in ("", " (%)"):
                if suffix and not (_unit(key) or h.endswith("(%)")):
                    continue
                hh = h if (not suffix or h.endswith("(%)")) else h + suffix
                if suffix == "" and h.endswith("(%)"):
                    continue
                if _keep_english(key) or e != e0:
                    en[hh] = e + (" (%)" if suffix and not e.endswith("%)") else "")
                es[hh] = s + (" (%)" if suffix and not s.endswith("%)") else "")
                # the builder may print the share without the word "share" ("Building vivienda ... o m s (%)"): translate that spelling too
                bare = re.sub(r" (share|pct)( \(%\))?$", lambda m_: m_.group(2) or "", hh)
                if bare != hh:
                    if _keep_english(key) or e != e0:
                        en.setdefault(bare, en.get(hh, e))
                    es.setdefault(bare, es[hh])
    # a plain ratio, not a percentage: the builder still appends " (%)" to the label it prints in its server-rendered lists
    for h in _builder_bases("ine_incomeP80P20Ratio"):
        h = h[:-4] if h.endswith(" (%)") else h
        en[h + " (%)"], es[h + " (%)"] = "Income P80/P20 ratio", "Ratio de renta P80/P20"
        en[h], es[h] = "Income P80/P20 ratio", "Ratio de renta P80/P20"
    return en, es


# ---- election columns (pycensus.countries.spain.elections): labels derived from the column key, so any party slug works.
# English: the builder's own label where it has one ("National election 2023: left bloc vote"); the columns it does not name (votes cast, electors,
# vote counts) get the same style here. Spanish: "Elecciones generales 2023: voto bloque izquierda".
_EL = _load("elections.json")
_ELEC_SCOPE_EN, _ELEC_SCOPE_ES = _EL["scope_en"], _EL["scope_es"]
_ELEC_BASE = {k: tuple(v) for k, v in _EL["base"].items()}   # what -> (English, Spanish)
_PARTY_NAMES = _EL["party_names"]
_ELEC_RE = re.compile(r"^(?:ine_)?elec_(national|municipal|state)_(\d{4})_(.+)$")


def party_name(slug: str) -> str:
    """Display name of a party slug (``eaj_pnv`` -> EAJ-PNV); unknown slugs: <= 4 letters upper case, else title case."""
    if slug in _PARTY_NAMES:
        return _PARTY_NAMES[slug]
    return slug.upper() if len(slug) <= 4 else " ".join(w.capitalize() for w in slug.split("_"))


def election_label(key: str, with_year: bool = False):
    """(English, Spanish, is_percent) of an ``elec_<scope>_<year>_<what>`` column key (``ine_`` prefix optional), or None."""
    m = _ELEC_RE.match(key)
    if not m:
        return None
    scope, year, what = m.groups()
    if what in _ELEC_BASE:
        we, ws = _ELEC_BASE[what]
    else:
        pm = re.match(r"^party_(.+)_(votes|share)$", what)
        if not pm:
            return None
        name = party_name(pm.group(1))
        we, ws = (f"{name} votes", f"votos {name}") if pm.group(2) == "votes" else (f"{name} vote", f"voto {name}")
    yr = (lambda t: t) if with_year else (lambda t: t.replace(" {y}", ""))
    return (f"{yr(_ELEC_SCOPE_EN[scope]).format(y=year)}: {we}", f"{yr(_ELEC_SCOPE_ES[scope]).format(y=year)}: {ws}", what.endswith("_share") or what == "turnout_rate")


def _multi_dated(columns):
    """Keys whose value exists for several dates in this build (only those keep a date in their name)."""
    g = {}
    for k in map(str, columns):
        m = _ELEC_RE.match(k)
        base = f"{m.group(1)}|{m.group(3)}" if m else re.sub(r"_(?:19|20)\d\d(?:M\d\d)?(?=_|$)", "", k)
        g.setdefault(base, []).append(k)
    return {k for l in g.values() if len(l) > 1 for k in l}


_DISC_ES = _load("scoring_axes_es.json")["strings"]
UI_ES.update(_DISC_ES)
UI_ES.update({f"{y} vs {x}": f"{_DISC_ES[y]} vs {_DISC_ES[x]}" for y in _DISC_ES for x in _DISC_ES})


def labels_for(columns):
    """Extra (English, Spanish) label dicts for the data columns of one build (election columns of any party; dated labels only for values present at several dates)."""
    en, es = {}, {}
    multi = _multi_dated(columns)
    for key in columns:
        key = str(key)
        r = election_label(key, with_year=key in multi)
        for h in _builder_bases(key):   # "... share (%)" / "... pct (%)" -> "... (%)": the % already says it is a share
            c = re.sub(r" (share|pct)( \(%\))?$", lambda m: m.group(2) or "", h)
            if c != h and r is None:
                en.setdefault(h, c)
        if r is None:
            if key in multi and key in COLUMNS:
                e0, s0 = COLUMNS[key]
                for h in _builder_bases(key):
                    en[h], es[h] = _PAREN_RE.sub("", e0).strip(), _PAREN_RE.sub("", s0).strip()
            continue
        e, s, pct = r
        for h in _builder_bases(key):
            h0 = h[:-4] if h.endswith(" (%)") else h
            es[h0] = s
            en[h0] = e
            if pct:
                es[h0 + " (%)"] = s + " (%)"
                en[h0 + " (%)"] = e + " (%)"
    return en, es


_SPANISH_RAW = re.compile(r"Vivienda|Hasta_|M_s_|Antes_de|No_consta|No_Consta|_a_\d")


def english_field_labels() -> dict:
    """English display names for the columns whose raw names are Spanish (INE building / age / floor-area bands), keyed by logical name (no ``ine_``):
    registered with the map builder so every list of the page prints English and the Spanish mode translates from it."""
    out = {}
    for key, (e, _s) in COLUMNS.items():
        if _SPANISH_RAW.search(key):
            out[key[4:] if key.startswith("ine_") else key] = simple(e)
            out[key] = simple(e)   # the full key too: some label functions look the column up before stripping its source prefix
    return out


try:   # before the label tables below: the builder's own labels (display_field_name) then already read English
    from transitlos.map import build as _b

    _b.register_field_labels(english_field_labels())
except Exception:  # noqa: BLE001 -- builder not importable (docs / tests)
    pass

COLUMN_LABELS_EN, COLUMN_LABELS_ES = _labels()
#: English-mode rewrites of the map's own emoji-prefixed layer labels (text only, no emoji).
UI_EN = dict(_load("ui_en.json")["strings"])
COLUMN_LABELS_EN.update(UI_EN)
