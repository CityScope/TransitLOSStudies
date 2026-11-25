"""Observed transit use per city on the finest unit any source publishes: the outcome table of the city science flex study.

    python flex/outcomes.py [city ...]          # every city by default -> <city>/results/flex_outcomes.parquet

One row per published unit (its own geometry, never split down): ``transit_share`` (0-1), ``n`` (commuters or trips the share is of, expanded
where the source expands), ``n_sample`` (people actually surveyed behind the share, where the source says; a census is its own sample),
``basis`` (what is counted, see BASIS), ``level``, ``source``. Shares of commuters are transit / (transit + car + walk + bike) as in
``code/modal_split.py``; the survey sources keep their own definition (Taipei counts every vehicle of a trip).

The fits use transit against the car only, as the US fit: ``share_tc`` = transit / (transit + car) (Taipei and Gipuzkoa: car = every private
motorised mode, motorcycles included, as their sources group them), ``n_tc`` / ``n_sample_tc`` = ``n`` / ``n_sample`` x (transit + car) share.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CACHE = str(HERE / "census")
FOUR = ["publicTransportCommuters", "carCommuters", "walkCommuters", "bikeCommuters"]
BASIS = {"commuters": "workers by usual main mode to work, by residence",
         "work_trips": "surveyed workers by main mode of their first work trip of the day, by residence",
         "trips_origin": "weekday trips of every purpose by mode, by the unit the trip starts in",
         "trips_residence": "residents' trips of every purpose by main mode, by residence",
         "vehicle_uses_commute": "vehicles used on residents' commuting / school trips (every vehicle of a trip counted), by residence",
         "respondents_day_before": "respondents' modes used the day before x residents, by residence"}


def _aoi(city: str) -> gpd.GeoDataFrame:
    return gpd.read_file(HERE / city / "aoi.gpkg").to_crs(4326)


def _commuters(g: gpd.GeoDataFrame, prefix: str = "", sample=None) -> gpd.GeoDataFrame:
    v = g[[prefix + c for c in FOUR]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    n = v.sum(1)
    out = gpd.GeoDataFrame({"GEOID": g["GEOID"].astype(str).to_numpy(), "transit_share": np.where(n > 0, v[:, 0] / np.maximum(n, 1e-9), np.nan),
                            "n": n, "n_sample": n if sample is None else sample}, geometry=g.geometry.to_numpy(), crs=g.crs)
    out = _tc(out, v[:, 0] / np.maximum(n, 1e-9), v[:, 1] / np.maximum(n, 1e-9))
    return out[out["n"] > 0]


def _tc(out, transit, car):
    """Adds share_tc = transit / (transit + car) and the counts behind it (n, n_sample x the transit + car share)."""
    t, c = np.asarray(transit, float), np.asarray(car, float)
    tc = t + c
    out["share_tc"] = np.where(tc > 0, t / np.where(tc > 0, tc, 1.0), np.nan)
    out["n_tc"] = pd.to_numeric(out["n"], errors="coerce").to_numpy(float) * tc
    out["n_sample_tc"] = pd.to_numeric(out["n_sample"], errors="coerce").to_numpy(float) * tc
    return out


def acs(city):
    g = gpd.read_parquet(HERE / city / "results" / "census" / "blockgroup.parquet")
    return _commuters(g), "block group", "commuters", "ACS 5-year"


def toronto(city):
    import glob

    parts = [gpd.read_parquet(f) for f in sorted(glob.glob(str(HERE / "census" / "aoi_*" / "canada" / "dissemination-area" / "35.parquet")))]
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).drop_duplicates("GEOID")
    g = g[g.intersects(_aoi(city).to_crs(g.crs).union_all())]
    return _commuters(g, "statcan_"), "dissemination area", "commuters", "Statistics Canada, Census 2021"


def concepcion(city):
    from pycensus.census.countries.chile.ine import manzana

    g = manzana.load(aoi=_aoi(city), data_dir=str(HERE / "census" / "chile"))
    return _commuters(g), "block (manzana) / rural entity", "commuters", "INE Chile, Censo 2024"


def beerseba(city):
    from pycensus.census.countries.israel import cbs_commute

    g = cbs_commute.load(aoi=_aoi(city), cache_dir=CACHE)
    return _commuters(g), "sub-quarter / locality", "commuters", "CBS Israel, Census 2022 public-use file"


def guadalajara(city):
    from pycensus.census.countries.mexico import imeplan_eod

    # the survey's coarsest zones (centralidades): at AGEB or microzone the ~16 surveyed commuters per unit are mostly noise, and the score is
    # not identified (flex/FIT_RESULTS.md, section 9)
    g = imeplan_eod.load(aoi=_aoi(city), level="centralidad", cache_dir=CACHE, data_dir=str(HERE / "census" / "mexico"))
    return (_commuters(g, sample=pd.to_numeric(g["imepeod_sampleCommuters"], errors="coerce").to_numpy()), "centralidad of residence", "work_trips",
            "IMEPLAN, Encuesta Origen-Destino 2023")


def gipuzkoa(city):
    """Donostia's 15 neighbourhoods (PMUS sheets) and, outside the city, the Basque mobility study's comarcas (all trips by origin, both)."""
    from pycensus.census.countries.spain.mobility import donostia_pmus, em_capv

    b = donostia_pmus.load(cache_dir=CACHE)
    nb = b["donpmus_tripsTotal"].astype(float)
    z = em_capv.load(aoi=_aoi(city), cache_dir=CACHE)
    z = z[z["GEOID"] != "20-CAP"]                     # the city itself comes from its neighbourhoods
    nz = z["emcapv_tripsTotal"].astype(float)
    out = gpd.GeoDataFrame({"GEOID": [*("auzoa-" + b["GEOID"]), *("zone-" + z["GEOID"])],
                            "transit_share": [*(b["donpmus_publicTransportTrips"] / nb), *(z["emcapv_publicTransportTrips"] / nz)],
                            "n": [*nb, *nz], "n_sample": np.nan,
                            "unit": [*(["neighbourhood"] * len(b)), *(["comarca"] * len(z))]},
                           geometry=[*b.geometry.to_crs(4326), *z.geometry.to_crs(4326)], crs=4326)
    car = [*((b["donpmus_carTrips"] + b["donpmus_motorcycleTrips"]) / nb), *((z["emcapv_carTrips"] + z["emcapv_motorcycleTrips"]) / nz)]
    out = _tc(out, out["transit_share"], car)
    return (out, "neighbourhood of Donostia (15) / comarca of the Basque mobility study", "trips_origin",
            "PMUS San Sebastian 2024-2029 (EM CAPV 2021 + 2022 oversample); Estudio de la Movilidad de la CAPV 2021")


def taipei(city):
    """Taipei City's 12 districts (city DOT survey 2018) and New Taipei's 7 traffic zones (MOTC survey 2010): vehicles used on commutes."""
    from pycensus.census.countries.taiwan import shuangbei_commute, taipei_mode_use

    g = taipei_mode_use.load(cache_dir=CACHE, data_dir=str(HERE / "census" / "taiwan"))
    z = shuangbei_commute.load(cache_dir=CACHE, data_dir=str(HERE / "census" / "taiwan"))
    z = z[z["county"] == "New Taipei City"]
    out = gpd.GeoDataFrame({"GEOID": [*("district-" + g["GEOID"]), *("zone-" + z["GEOID"])],
                            "transit_share": [*g["transit_commute_share"], *z["transit_commute_share"]], "n": np.nan,
                            "n_sample": [*g["sample_out_yesterday"], *z["sample_trips"]],
                            "year": [*([2018] * len(g)), *([2010] * len(z))]},
                           geometry=[*g.geometry, *z.geometry], crs=4326)
    out = _tc(out, out["transit_share"], [*g["private_motorised_commute_share"], *z["private_motorised_commute_share"]])
    return (out, "district of Taipei City (12) / traffic zone of New Taipei (7)", "vehicle_uses_commute",
            "Taipei DOT daily mode-use survey 2018; MOTC daily mode-use survey 2010 (Taipei DOT report 102-03)")


def hamburg(city):
    """Hamburg's 7 Bezirke (MiD 2023) and the Kreise of the Metropolregion around it (MiD 2017 regional report)."""
    from pycensus.census.countries.germany import mid_hamburg

    b = mid_hamburg.bezirke(cache_dir=CACHE)
    k = mid_hamburg.kreise(data_dir=str(HERE / "census" / "germany"))
    out = gpd.GeoDataFrame({"GEOID": [*("bezirk-" + b["GEOID"]), *("kreis-" + k["GEOID"])],
                            "transit_share": [*b["transit_trip_share"], *k["transit_trip_share"]], "n": np.nan, "n_sample": np.nan,
                            "year": [*([2023] * len(b)), *([2017] * len(k))]},
                           geometry=[*b.geometry.to_crs(4326), *k.geometry.to_crs(4326)], crs=4326)
    out = _tc(out, out["transit_share"], [*b["car_trip_share"], *k["car_trip_share"]])
    return out, "Bezirk of Hamburg (7) / Kreis of the Metropolregion (13)", "trips_residence", "MiD 2023 Hamburg; MiD 2017 Metropolregion Hamburg regional report"


def andorra(city):
    sys.path.insert(0, str(HERE / "code"))
    import modal_split

    return _commuters(modal_split._andorra_units(), sample=np.nan), "parish", "respondents_day_before", "ENM 2050 citizen survey (CRES 2021)"


CITIES = {"boston": acs, "san_francisco": acs, "toronto": toronto, "concepcion": concepcion, "beerseba": beerseba, "guadalajara": guadalajara,
          "gipuzkoa": gipuzkoa, "taipei": taipei, "hamburg": hamburg, "andorra": andorra}
# shanghai: no mode split below the city is published


def build(city: str) -> gpd.GeoDataFrame:
    """The outcome table of one city, written to ``<city>/results/flex_outcomes.parquet``.

    Args:
        city: City folder name.

    Returns:
        The table.
    """
    g, level, basis, source = CITIES[city](city)
    g = g.to_crs(4326).assign(city=city, level=level, basis=basis, basis_text=BASIS[basis], source=source)
    g.to_parquet(HERE / city / "results" / "flex_outcomes.parquet")
    return g


if __name__ == "__main__":
    for c in sys.argv[1:] or list(CITIES):
        try:
            t = build(c)
            w = t["n"].fillna(1.0)
            print(f"{c:14s} {len(t):6d} {t['level'].iloc[0]:32s} {t['basis'].iloc[0]:22s} transit share {np.average(t['transit_share'], weights=w):.3f}"
                  f"  sample {t['n_sample'].sum():,.0f}", flush=True)
        except Exception as exc:
            print(f"{c:14s} FAILED: {type(exc).__name__}: {exc}", flush=True)
