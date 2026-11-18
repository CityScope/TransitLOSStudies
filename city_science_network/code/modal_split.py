"""Modal split (transit / walk / car shares) per map area of every city-science map: `<city>/results/modal_split.json`
{area: {"transit_share", "walk_share", "car_share", "basis", "source", "level"}} for the areas of the place-ranking tab
(metro = UN metro, core = UN core, official = official metro, municipality).

- Units as PUBLISHED, each assigned whole to an area by its representative point; area share = transit commuters / commuters by car,
  public transport, walking and cycling summed over the area (only upscaling, never a value split down):
  block groups (Boston, San Francisco, ACS 5-year), dissemination areas (Toronto, StatCan 2021), census districts (Concepcion, INE Censo
  2024), CBS Census 2022 sub-quarters / localities (Beersheba, pycensus israel.cbs_commute), INEGI Censo 2020 municipalities (Guadalajara,
  pycensus mexico.inegi_movilidad), parishes (Andorra, ENM 2050 survey, pycensus andorra.enm_mobility: respondents' modes x parish residents).
- Gipuzkoa: INE ECEPOV-2021 (pycensus spain.mobility.ecepov): Donostia and the province.
- Published city figures (pycensus germany.mid_hamburg, taiwan.motc_mode_use): Hamburg (MiD 2023), Taipei (MOTC 2022).
- Shanghai (pycensus china.shanghai_transport_survey): only the central city's commuting share exists, a territory none of the areas matches.

    python code/modal_split.py [city ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
AREAS = {"metro": "area_un_metro", "core": "area_un_core", "official": "area_official_metro", "municipality": "area_municipality"}
CENSUS = {"boston": "ACS 5-year (commuters by mode)", "san_francisco": "ACS 5-year (commuters by mode)", "toronto": "Statistics Canada, Census 2021 (commuters by main mode)",
          "concepcion": "INE Chile, Censo 2024 (commuters by mode)"}
def _published() -> dict:
    """Municipality value + the value of the rest of its region (both published); every area = population-weighted mix of the two."""
    from pycensus.countries.germany import mid_hamburg
    from pycensus.countries.taiwan import motc_mode_use

    c = str(HERE.parent / "cache")
    h, t = mid_hamburg.load(c), motc_mode_use.load(c)
    return {
        "hamburg": {"muni": h["hamburg"]["transit_share"], "rest": h["hvv_umland"]["transit_share"], "basis": h["hamburg"]["basis"],
                    "source": h["hamburg"]["source"] + "; " + h["hvv_umland"]["source"]},
        "taipei": {"muni": t["taipei_city"]["transit_share"], "rest": t["new_taipei_city"]["transit_share"], "basis": t["taipei_city"]["basis"],
                   "source": t["taipei_city"]["source"] + "; New Taipei City 28.7 %"},
    }


PUBLISHED = ("hamburg", "taipei")


AREA_FILES = {"metro": "un_metro.gpkg", "official": "metro_official.gpkg", "municipality": "municipality.gpkg", "core": "core.gpkg"}
UNITS = {   # the finest unit the census PUBLISHES commute modes for (never a value split down to smaller units or cells)
    "boston": ("results/census/blockgroup.parquet", ""), "san_francisco": ("results/census/blockgroup.parquet", ""),
    "concepcion": ("results/census/distrito.parquet", ""),   # districts = sums of the published census blocks (manzanas nest in them)
    "toronto": ("census/aoi_*/canada/dissemination-area/35.parquet", "statcan_"),
}
LOADED = {   # units read through a pycensus source: city -> (source label, census label, loader(aoi) -> GeoDataFrame with the four counts)
    "beerseba": ("CBS Israel, Census 2022 public-use file (main means of transport to work)", "sub-quarters / localities",
                 lambda aoi: __import__("pycensus.countries.israel.cbs_commute", fromlist=["load"]).load(aoi=aoi, cache_dir=str(HERE / "census"))),
    "guadalajara": ("INEGI, Censo 2020 cuestionario ampliado (means of transport to work)", "municipalities",
                    lambda aoi: __import__("pycensus.countries.mexico.inegi_movilidad", fromlist=["load"]).load(
                        aoi=aoi, cache_dir=str(HERE / "census"), data_dir=str(HERE / "census" / "mexico"))),
    "andorra": ("ENM 2050 citizen survey (CRES 2021): modes used the day before, by parish", "parishes", None),
}
MODE_COLS = ["publicTransportCommuters", "carCommuters", "walkCommuters", "bikeCommuters"]


def _area_union(city: str):
    import geopandas as gpd

    gs = [gpd.read_file(HERE / city / f).to_crs(4326) for f in AREA_FILES.values() if (HERE / city / f).is_file()]
    return gpd.GeoDataFrame(geometry=[pd.concat(gs).union_all()], crs=4326)


def _andorra_units():
    import geopandas as gpd

    from pycensus.countries.andorra import enm_mobility

    t = enm_mobility.parish_table(str(HERE / "census"))
    g = gpd.read_parquet(HERE / "andorra" / "results" / "census" / "parish.parquet", columns=["GEOID", "geometry", "census_residents"])
    g = g.merge(t, on="GEOID")
    pop = pd.to_numeric(g["census_residents"], errors="coerce").fillna(0.0)
    for col, share in zip(MODE_COLS, ("transit_share", "car_share", "walk_share", "bike_share")):
        g[col] = pop * g[share]   # residents x share of respondents of the parish using the mode
    return g


def _units(city: str):
    import glob

    import geopandas as gpd

    pat, pre = UNITS[city]
    files = sorted(glob.glob(str(HERE / city / pat))) or sorted(glob.glob(str(HERE / pat)))
    cols = [pre + c for c in MODE_COLS]
    parts = [gpd.read_parquet(f, columns=["GEOID", "geometry"] + cols) for f in files]
    u = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).drop_duplicates("GEOID")
    return u.rename(columns=dict(zip(cols, MODE_COLS)))


def _from_cells(city: str) -> dict:
    """Census units published at their own level, each assigned WHOLE to an analysis area by its representative point; area share = sum of
    transit commuters / sum of commuters by car, transit, walking and cycling (an exact sum of published counts: upscaling only)."""
    import geopandas as gpd
    import shapely

    if city in LOADED:
        u = (_andorra_units() if city == "andorra" else LOADED[city][2](_area_union(city))).to_crs(4326)
    else:
        u = _units(city).to_crs(4326)
    pts = u.geometry.representative_point()
    cols = ["publicTransportCommuters", "walkCommuters", "carCommuters", "bikeCommuters"]
    v = u[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = {}
    for area, f in AREA_FILES.items():
        p = HERE / city / f
        if not p.is_file():
            continue
        g = gpd.read_file(p).to_crs(4326).union_all()
        inside = shapely.contains_xy(g, pts.x.to_numpy(), pts.y.to_numpy())
        if not inside.any():
            continue
        s = v[inside].sum()
        tot = float(s.sum())
        if tot <= 0:
            continue
        out[area] = {"transit_share": round(s["publicTransportCommuters"] / tot, 4), "walk_share": round(s["walkCommuters"] / tot, 4),
                     "car_share": round(s["carCommuters"] / tot, 4), "bike_share": round(s["bikeCommuters"] / tot, 4),
                     "basis": "residents' trips the day before (car, public transport, walking, cycling)" if city == "andorra" else "commuters (car, public transport, walking, cycling)",
                     "source": LOADED[city][0] if city in LOADED else CENSUS[city],
                     "level": f"{int(inside.sum()):,} " + (LOADED[city][1] if city in LOADED else f"census units ({UNITS[city][0].split('/')[-1].split('.')[0]})") + " by representative point"}
    return out


def _mixed(city: str, s_muni: float, s_rest: float, basis: str, source: str, only=None) -> dict:
    """Each area's share = population-weighted mix of the municipality's value (its cells inside the municipality) and the rest of the
    region's value (the other cells), from the cells' census population."""
    f = HERE / city / "results" / "downloads" / "h3_res9.parquet"
    d = pd.read_parquet(f)
    pop = pd.to_numeric(d.get("census_residents", d.get("population")), errors="coerce").fillna(0.0)
    inm = d["area_municipality"].fillna(False).astype(bool)
    out = {}
    for area, flag in AREAS.items():
        if only and area not in only:
            continue
        m = d[flag].fillna(False).astype(bool) if flag in d.columns else None
        if m is None or not m.any():
            continue
        p_in, p_out = float(pop[m & inm].sum()), float(pop[m & ~inm].sum())
        if p_in + p_out <= 0:
            continue
        share = (p_in * s_muni + p_out * s_rest) / (p_in + p_out)
        out[area] = {"transit_share": round(share, 4), "basis": basis, "source": source,
                     "level": "municipality" if area == "municipality" else f"mix: {100 * p_in / (p_in + p_out):.0f}% of residents in the municipality, the rest at the region's value"}
    return out


def _gipuzkoa() -> dict:
    ms = json.loads((HERE.parent / "spain_study" / "data" / "modal_split_ecepov2021.json").read_text(encoding="utf-8"))
    src = "INE ECEPOV-2021 (persons 16+, main means of transport to work or study)"
    pick = lambda e, lvl: {"transit_share": e["transit_share"], "walk_share": e["walk_share"], "car_share": e["car_share"],  # noqa: E731
                           "basis": "trips to work or study", "source": src, "level": lvl}
    out = {}
    don, prov = ms["muni"].get("20069"), ms["prov"].get("20")
    if don and prov:   # UN metro / UN core: Donostia + the rest of the province (province counts minus Donostia's), mixed by population
        rest = (prov["public"] - don["public"]) / max(prov["total"] - don["total"], 1)
        out.update(_mixed("gipuzkoa", don["transit_share"], rest, "trips to work or study", src + ": Donostia, and the rest of Gipuzkoa (province minus Donostia)",
                          only=("metro", "core")))
    if don:
        out["municipality"] = pick(don, "Donostia / San Sebastián")
    if prov:
        out["official"] = pick(prov, "province of Gipuzkoa")
    return out


def build(city: str) -> Path | None:
    if city in CENSUS or city in LOADED:
        data = _from_cells(city)
    elif city == "gipuzkoa":
        data = _gipuzkoa()
    elif city in PUBLISHED:
        c = _published()[city]
        data = _mixed(city, c["muni"], c["rest"], c["basis"], c["source"])
    else:
        data = {}
    if not data:
        return None
    p = HERE / city / "results" / "modal_split.json"
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(city, {a: v["transit_share"] for a, v in data.items()})
    return p


if __name__ == "__main__":
    cities = sys.argv[1:] or ["andorra", "beerseba", "boston", "concepcion", "guadalajara", "hamburg", "shanghai", "taipei", "toronto", "san_francisco", "gipuzkoa"]
    for c in cities:
        build(c)
