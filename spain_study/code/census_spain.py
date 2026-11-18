# -*- coding: utf-8 -*-
"""Section-level census assembly for one Spanish province, with the SAME column names as city_science_network (prefix ``ine_``).

Mirrors ``city_science_network/code/pipeline.py::_spain_census_loader`` + ``_spain_housing_extras`` (not imported; copied pattern):
  * INE Padron population per section (``ine.load``)                        -> ``population`` + ``ine_*``
  * Census 2021 (``censo2021.load``) and INE ADRH income atlas (``adrh``)   -> ``ine_<bare name>``
  * SERPAVI rent, INE tourist dwellings, net migration, MIVAU valor tasado  -> ``ine_serpavi_*``, ``ine_tourist_*``, ``ine_netmig_*``, ``ine_valor_tasado_*``
  * Census 2021 housing (district-native tenure / year built / floor area / building shares, repeated on sections;
    ``household_size_mean`` is all NaN and dropped)                           -> ``ine_<name>``
  * Catastro buildings aggregated to sections (state INSPIRE, Gipuzkoa B5M; pycensus ``catastro``)  -> ``ine_catastro_*``
  * Basque Country: Eustat/EMAL rent, income and dwellings                   -> ``ine_emal_*``, ``ine_eustat_*``

Every optional source is wrapped: a failure prints a warning and keeps the rest.

RATES vs COUNTS (``is_rate``): rates (means, medians, shares, prices per m2, indices) are AVERAGED (population-weighted) when
areas are merged or cells aggregated; counts (population, dwellings, households, tourist dwellings, ...) are SUMMED, and are
apportioned by area/population share when a section is split between H3 cells. Same rule as the city studies' ``_is_rate_col``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Counts whose names contain a rate word above (explicit exceptions, as ``_SPAIN_COUNT_EXTRAS`` in the city studies).
COUNT_EXCEPTIONS = {
    "ine_serpavi_n_vc_2024", "ine_dwellings", "ine_eustat_dwellingsTotal", "ine_eustat_dwellingsFamily", "ine_eustat_dwellingsMain",
    "ine_eustat_dwellingsOther", "ine_emal_deposits",
}
#: Rates whose names carry no rate word (prices, tenure fractions, ...).
EXTRA_RATE_PREFIXES = ("ine_serpavi_rent", "ine_serpavi_area", "ine_valor_tasado", "ine_emal_rent", "ine_tenure_", "ine_built_", "ine_floor_area_",
                       "ine_building_", "ine_mitma_", "ine_catastro_dwellings_per_building", "ine_dwellings_main_share", "ine_dwellings_secondary", "ine_eustat_income")
#: Municipality natives broadcast to sections: totals are NOT apportioned (they describe the whole municipality) -> treated as rates for mapping.
MUNICIPALITY_NATIVE_PREFIXES = ("ine_tourist_", "ine_netmig_", "ine_valor_tasado", "ine_emal_")


def is_rate(col: str) -> bool:
    """True if ``col`` is intensive (averaged) rather than additive (summed)."""
    if col.startswith("ine_mitma_regular_flows_") and "share" not in col:   # person-pair counts split among sections: additive
        return False
    if col.startswith("ine_elec_"):   # explicit (party slugs can contain rate-like words): shares and turnout are rates, votes / electors are counts
        return col.endswith(("_share", "_turnout_rate"))
    if col in COUNT_EXCEPTIONS:
        return False
    if col.startswith(MUNICIPALITY_NATIVE_PREFIXES) or col.startswith(EXTRA_RATE_PREFIXES):
        return True
    low = col.lower()
    return any(s in low for s in ("mean", "median", "rate", "ratio", "share", "density", "gini", "p80", "p20", "index", "avg", "pct"))


#: Columns the source publishes on a 0-100 scale; the map's shared builder shows every ``*share`` / ``*pct`` column as a 0-1 fraction x 100 (``%``).
PERCENT_100_RE = re.compile(r"^ine_(incomeSource.*Share|singlePersonHouseholdsShare|aboveMedianShare\d+|below(Half)?MedianShare\d+|tourist_dwellings_share_.*|eustat_.*Pct)$")


SHARE_RE = re.compile(r"(?i)share|turnout_rate")
#: Prices / incomes / rents: a negative value is a data error (NaN). Not net migration, which is legitimately negative.
POSITIVE_RE = re.compile(r"(?i)income|rent|valor_tasado|eur_m2|price")
KEEP_ALWAYS = {"GEOID", "geometry", "h3", "h3_cell", "population", "level_of_service"}


def map_columns(df: pd.DataFrame, report: Optional[dict] = None) -> pd.DataFrame:
    """Map-only view of a table (applied ONLY to what is handed to ``build_city_map`` / the stats chunks, never to the stored pipeline outputs).

    * 0-100 percentages rescaled to 0-1 (``PERCENT_100_RE``), then EVERY share / turnout column clipped to [0, 1] (a part never exceeds its whole);
    * negative price / income / rent values -> NaN;
    * the duplicate ``ine_population`` (== ``population``) dropped;
    * numeric columns that are entirely NaN dropped (names collected in ``report["dropped"]``; clip / negative counts in ``report["clipped"]`` /
      ``report["negative"]``).
    """
    df = df.copy()
    report = report if report is not None else {}
    from pycensus.global_equivalents import add_global_equivalents

    add_global_equivalents(df)   # period-free global twins (elec_national_left_share, rent_per_m2_mean, ...) of the latest period
    num = [c for c in df.columns if c not in KEEP_ALWAYS and pd.api.types.is_numeric_dtype(df[c])]
    for c in num:
        if PERCENT_100_RE.match(c):
            df[c] = df[c] / 100.0
        if SHARE_RE.search(c):
            over = int(((df[c] > 1) | (df[c] < 0)).sum())
            if over:
                report.setdefault("clipped", {})[c] = over
            df[c] = df[c].clip(0, 1)
        elif POSITIVE_RE.search(c):
            neg = int((df[c] < 0).sum())
            if neg:
                report.setdefault("negative", {})[c] = neg
                df.loc[df[c] < 0, c] = np.nan
    if "population" in df.columns and "ine_population" in df.columns:
        df = df.drop(columns="ine_population")
    empty = [c for c in num if c in df.columns and df[c].isna().all()]
    if empty:
        report.setdefault("dropped", []).extend(empty)
        df = df.drop(columns=empty)
    # Only the LAST election per scope and the latest-period housing / rent / price / tourist / migration columns are shown (stored parquet keeps all).
    from pycensus.latest_columns import latest_column_filter

    _, historic = latest_column_filter(df.columns, protect=[r"^(GEOID|geometry)$"])
    if historic:
        report["historic_dropped"] = historic
        df = df.drop(columns=list(historic))
    return df


def with_native_columns_as_relative(fn):
    """``is_relative_field`` replacement: municipality natives (and every other ``is_rate`` column) count as relative.

    Municipality-native columns repeat on every section, so summing them (circle size, distribution bars) is wrong; as relative fields
    they are offered for opacity / colour / regression / ANOVA only.
    """
    def wrapped(col):
        c = str(col)
        return fn(col) or c.startswith(MUNICIPALITY_NATIVE_PREFIXES) or (c.startswith("ine_") and is_rate(c))
    return wrapped


def split_columns(df: pd.DataFrame):
    num = [c for c in df.columns if c not in ("GEOID", "geometry", "level_of_service") and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in num if is_rate(c)], [c for c in num if not is_rate(c)]


def _attach(result, frame, mapping, on_mun=False):
    keep = frame[["GEOID", *[c for c in mapping if c in frame.columns]]].rename(columns=mapping)
    if on_mun:
        keep = keep.rename(columns={"GEOID": "_mun"})
        return result.assign(_mun=result["GEOID"].astype(str).str[:5].to_numpy()).merge(keep, on="_mun", how="left").drop(columns="_mun")
    return result.merge(keep, on="GEOID", how="left")


_MITMA_CACHE: dict = {}


def mitma_table(cache_dir: str) -> pd.DataFrame:
    """MITMA distance / long-distance / regular-commute columns (``ine_mitma_*``, year dropped) indexed by 10-digit section id."""
    from pycensus.countries.spain.mobility import mitma

    if "t" not in _MITMA_CACHE:
        trips = mitma.load_sections(2024, cache_dir=cache_dir)
        comm = mitma.load_commuting(2024, cache_dir=cache_dir)
        t = trips.join(comm)
        t.columns = [f"ine_mitma_{c[5:]}" for c in t.columns]   # drop the "2024_" prefix
        t.columns = [c.replace("_>50", "_over50").replace("0.5-2", "0_5_2").replace("2-10", "2_10").replace("10-50", "10_50").replace("1-2", "1_2").replace("3-4", "3_4").replace("5-7", "5_7").replace("8-10", "8_10").replace("11-13", "11_13") for c in t.columns]
        _MITMA_CACHE["t"] = t
    return _MITMA_CACHE["t"]


def mobility_extras(result: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    """Join the MITMA table on the 10-digit section id."""
    t = mitma_table(cache_dir)
    return result.merge(t, left_on=result["GEOID"].astype(str).str.zfill(10).to_numpy(), right_index=True, how="left").drop(columns="key_0", errors="ignore")


def housing_extras(result: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Same joins and ``ine_`` names as ``_spain_housing_extras`` (section level)."""
    from pycensus.countries.spain import serpavi, valor_tasado
    from pycensus.countries.spain.ine_tables import load_migration_balance, load_tourist_dwellings

    provinces = sorted({str(g)[:2] for g in result["GEOID"]})
    try:
        sp = serpavi.load_attributes(provinces, "section", data_dir)
        mapping = {}
        for y in (2024, 2016, 2011):
            mapping[f"serpavi_rent_m2_median_vc_{y}"] = f"ine_serpavi_rent_m2_median_vc_{y}"
        for y in (2024, 2016):
            mapping[f"serpavi_rent_total_median_vc_{y}"] = f"ine_serpavi_rent_total_median_vc_{y}"
        for name in ("rent_m2_median_vu_2024", "rent_m2_p25_vc_2024", "rent_m2_p75_vc_2024", "area_m2_median_vc_2024", "n_vc_2024", "rent_m2_latest"):
            mapping[f"serpavi_{name}"] = f"ine_serpavi_{name}"
        result = _attach(result, sp, mapping)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SERPAVI rent join skipped: {exc}")
    try:
        tour = load_tourist_dwellings(provinces, data_dir)
        result = _attach(result, tour, {f"ine_tourist_{k}_{t}": f"ine_tourist_{k}_{t}" for k in ("dwellings", "places", "dwellings_share") for t in ("2020M08", "2026M05")}, True)
        mig = load_migration_balance(provinces, data_dir)
        result = _attach(result, mig, {f"ine_netmig_{k}_2024": f"ine_netmig_{k}_2024" for k in ("total", "exterior", "interior")}, True)
        vt = valor_tasado.load_attributes(data_dir)
        result = _attach(result, vt, {f"valor_tasado_{k}_T2A{y}": f"ine_valor_tasado_{k}_eur_m2_{y}" for k in ("total", "new", "old") for y in (2026, 2016)}, True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"municipal housing / tourism joins skipped: {exc}")
    try:
        from pycensus.countries.spain.censo2021.housing import load_section_housing

        ch = load_section_housing(data_dir).drop(columns=["district", "household_size_mean"], errors="ignore")
        result = _attach(result, ch, {c: f"ine_{c}" for c in ch.columns if c != "GEOID"})
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Census 2021 housing join skipped: {exc}")
    for prov in provinces:   # Catastro buildings (state INSPIRE; Gipuzkoa foral B5M); Araba / Bizkaia / Navarra not wired yet
        if "geometry" not in result.columns:
            break
        try:
            from pycensus.countries.spain import catastro

            mask = result["GEOID"].astype(str).str.startswith(prov)
            ca = catastro.load_attributes(prov, result.loc[mask, ["GEOID", "geometry"]], data_dir)
            ca = ca.dropna(axis=1, how="all")   # e.g. Gipuzkoa publishes no dwellings / use / floor area
            result = _attach(result, ca, {c: f"ine_{c}" for c in ca.columns if c != "GEOID"})
        except NotImplementedError as exc:
            logger.info(f"Catastro skipped for {prov}: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Catastro join skipped for {prov}: {exc}")
    result = elections_join(result, provinces, data_dir)
    basque = [p for p in provinces if p in ("01", "20", "48")]
    if basque:
        try:
            from pycensus.countries.euskadi.eustat import housing_income as hi

            em = hi.load_emal_municipal(2025, data_dir, tuple(basque))
            em16 = hi.load_emal_municipal(2016, data_dir, tuple(basque)).rename(columns={"emalRentM2": "emalRentM2_2016"})
            em = em.merge(em16[["GEOID", "emalRentM2_2016"]], on="GEOID", how="left")
            result = _attach(result, em, {"emalRentM2": "ine_emal_rent_m2_2025", "emalRentM2_2016": "ine_emal_rent_m2_2016",
                                          "emalRentMonthly": "ine_emal_rent_monthly_2025", "emalDeposits": "ine_emal_deposits"}, True)
            inc = hi.load_income_sections(tuple(basque), data_dir)
            result = _attach(result, inc, {c: f"ine_eustat_{c}" for c in inc.columns if c.startswith("income")})
            if "20" in basque:
                dw = hi.load_dwellings_sections(data_dir)
                result = _attach(result, dw, {c: f"ine_eustat_{c}" for c in dw.columns if c.startswith("dwelling")})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Basque housing / income join skipped: {exc}")
    return result


def _region(province: str) -> str:
    """pycensus region token of a province: codes 01..19 collide with autonomous-community codes, so they are passed as ``prov<NN>``."""
    return f"prov{province}" if int(province) < 20 else province


def elections_join(result, provinces, data_dir):
    """Congreso 2023, municipal 2023 (mesa files cover only municipalities above 250 inhabitants: NaN elsewhere) and, for 01 / 20 / 48,
    Parlamento Vasco 2024, at section level -> ``ine_elec_*`` (``..._votes`` / ``..._electors`` counts, shares and turnout rates)."""
    try:
        from pycensus.countries.spain import elections
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"elections skipped: {exc}")
        return result
    basque = [p for p in provinces if p in ("01", "20", "48")]
    jobs = [("congreso 2023", elections.load_congreso_2023, list(provinces)), ("municipal 2023", elections.load_municipal_2023, list(provinces))]
    if basque:
        jobs.append(("parlamento vasco 2024", elections.load_parlamento_vasco_2024, basque))
    for name, fn, provs in jobs:
        try:
            el = fn(level="section", provinces=provs, data_dir=data_dir)
            el = el.dropna(axis=1, how="all")
            result = _attach(result, el, {c: f"ine_{c}" for c in el.columns if c.startswith("elec_")})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"elections {name} skipped for {provs}: {exc}")
    return result


def load_province_sections(province: str, data_dir: str, cache_dir: str) -> gpd.GeoDataFrame:
    """One row per census section of ``province`` (2-digit code): geometry (4326), ``population`` and all ``ine_*`` columns."""
    from pycensus.countries.spain import adrh, censo2021, ine

    base = ine.load(regions=[_region(province)], level="section", cache_dir=cache_dir, data_dir=data_dir)
    # Population: Padron's bare/ine_ population column is exposed as ``population`` (this study's activity column).
    if "population" not in base.columns and "ine_population" in base.columns:
        base["population"] = base["ine_population"]
    if "ine_population" not in base.columns and "population" in base.columns:
        base["ine_population"] = base["population"]
    # ine.load returns bare names for canonical columns; prefix every non-id column with ine_ (city-study convention)
    ren = {c: f"ine_{c}" for c in base.columns if c not in ("GEOID", "geometry", "population") and not c.startswith("ine_")}
    base = base.rename(columns=ren)
    try:
        extra = censo2021.load(regions=[_region(province)], level="section", cache_dir=cache_dir, data_dir=data_dir)
        extra = extra.drop(columns=["geometry"]).rename(columns=lambda c: c if c == "GEOID" or c.startswith("ine_") else f"ine_{c}")
        extra = extra[[c for c in extra.columns if c == "GEOID" or c not in base.columns]]
        base = base.merge(extra, on="GEOID", how="left")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"censo2021 join skipped: {exc}")
    try:
        attrs = adrh.load_attributes([province], "section", data_dir=data_dir)
        if province in ("01", "20", "48"):
            from pycensus.countries.spain.adrh.eustat_sections import load_sections

            eus = load_sections([province], data_dir=data_dir)
            if len(eus):
                attrs = attrs.set_index("GEOID").combine_first(eus.set_index("GEOID")).reset_index()
        attrs = attrs.rename(columns=lambda c: c if c == "GEOID" or c.startswith("ine_") else f"ine_{c}")
        attrs = attrs[[c for c in attrs.columns if c == "GEOID" or c not in base.columns]]
        base = base.merge(attrs, on="GEOID", how="left")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"ADRH join skipped: {exc}")
    base = housing_extras(base, data_dir)
    try:
        base = mobility_extras(base, cache_dir)
    except Exception as exc:  # noqa: BLE001 - MITMA columns are optional
        logger.warning(f"MITMA mobility join skipped: {exc}")
    base["municipality_id"] = base["GEOID"].astype(str).str[:5]
    base["province_id"] = province
    return gpd.GeoDataFrame(base, geometry="geometry", crs=4326)


def column_kinds(df: pd.DataFrame) -> dict:
    """``{column: "rate" | "count"}`` for every numeric census column (written next to the data as ``column_kinds.json``)."""
    rates, counts = split_columns(df)
    return {**{c: "rate" for c in rates}, **{c: "count" for c in counts}}


def write_column_kinds(df: pd.DataFrame, path: Path) -> None:
    Path(path).write_text(json.dumps(column_kinds(df), indent=1, ensure_ascii=False))
