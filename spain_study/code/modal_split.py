"""Commuting modal split in Spain from INE's ECEPOV-2021 (Encuesta de Características Esenciales de la Población y las Viviendas):
persons aged 16+ by the main means of transport to their place of work or study.

- municipalities of more than 50,000 inhabitants and provincial capitals (151): table 55377
- provinces (52): table 55371

Writes `data/modal_split_ecepov2021.json`: {"muni": {INE code: {...}}, "prov": {code: {...}}, "source": ...}, each entry with the
counts (total, public, private, walk, other) and the shares transit / car (private vehicle) / walk. Municipality names are matched to INE
codes through the study's own municipality names (`_spain/stats_chunks/index.json`), article-normalised ("Ejido, El" = "El Ejido").

    python spain_study/code/modal_split.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE.parent / "cache" / "ine_ecepov"
from pycensus.countries.spain.mobility import ecepov as _ec

_norm = _ec.norm_name


def _table(t):
    return _ec._table(t, str(HERE.parent / "cache"))


_entries = _ec._entries
TABLES = _ec.TABLES


def build() -> Path:
    index = json.loads((HERE / "data" / "_spain" / "stats_chunks" / "index.json").read_text())
    munis, provs = {}, {}
    for st in index["states"]:
        for p in st["places"]:
            munis.setdefault(_norm(p["name"]), []).append((p["id"], p.get("pop", 0)))
        for c in st["counties"]:
            provs[_norm(c["name"])] = c["id"]
    out = {"source": "INE, ECEPOV-2021 (tables 55377 municipalities, 55371 provinces): persons 16+ by main means of transport to work or study",
           "muni": {}, "prov": {}, "unmatched": []}
    for name, e in _entries(_table(TABLES["muni"]), "Municipios").items():
        cands = munis.get(_norm(name), []) or [c for part in re.split(r"[/-]", name) for c in munis.get(_norm(part), [])]
        if not cands:
            out["unmatched"].append(name)
            continue
        code = max(cands, key=lambda c: c[1])[0]   # same name twice: the populous one (the survey only covers > 50,000 or capitals)
        out["muni"][code] = {"name": name, **e}
    alias = {"araba alava": "alava", "alava": "araba alava", "valencia valencia": "valencia", "castellon castello": "castellon", "alicante alacant": "alicante",
             "balears illes": "baleares islas baleares", "a coruna": "la coruna", "coruna a": "a coruna", "palmas las": "las palmas", "rioja la": "la rioja", "gipuzkoa": "gipuzkoa", "bizkaia": "bizkaia"}
    for name, e in _entries(_table(TABLES["prov"]), "Provincias").items():
        n = _norm(name)
        code = provs.get(n) or next((provs[_norm(part)] for part in name.split("/") if _norm(part) in provs), None) or next((v for k, v in provs.items() if set(k.split()) == set(n.split())), None)
        code = code or provs.get(alias.get(n, "")) or next((v for k, v in provs.items() if k.startswith(n.split()[0]) and len(n.split()[0]) > 3), None)
        if name.startswith("Total"):
            out["national"] = e
        elif code:
            out["prov"][code] = {"name": name, **e}
        else:
            out["unmatched"].append(name)
    path = HERE / "data" / "modal_split_ecepov2021.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"modal split: {len(out['muni'])} municipalities, {len(out['prov'])} provinces; unmatched: {out['unmatched']}")
    return path


if __name__ == "__main__":
    build()
