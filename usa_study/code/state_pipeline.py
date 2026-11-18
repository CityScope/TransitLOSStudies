# -*- coding: utf-8 -*-
"""Stage 3: state-level rollup -- county.geoparquet + coarser h3 resolutions.

Per `usa_study/prompt.txt`: "Lastly for the complete state do the
county.geoparquet file and the rest of h3 resolutions." Counties don't
overlap, so the county-level rollup is a plain concat of every county's
own single-row `county.geoparquet`, and h3 res 9/11 are a plain concat of
every county's own h3 outputs (no re-aggregation needed, since the
per-county grids are disjoint by construction -- built from
non-overlapping county boundaries). Coarser resolutions (5, 7 -- matching
`city_science_network`'s own `map_h3_resolutions` zoom-banded set) are
new at the state level and get resampled UP from the concatenated res-9
table: population/jobs are summed (a real total), `level_of_service` is
population-weighted (never averaged unweighted, which would let a
near-zero-population cell distort a coarser cell's real access
experience).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Matches every `{year}_{office}_...` election column -- kept in sync with
#: `county_pipeline._ELECTION_COL_RE` (same naming convention, from
#: `elections.schema`).
_ELECTION_COL_RE = re.compile(r"^\d{4}_(president|senate)_")

#: State-level h3 resolutions, matching `city_science_network.code.params
#: .StudyParams.map_h3_resolutions`'s own zoom-banded set. 9 and 11 are
#: 11 (LOS>0 or dense tracts), 9 and 7 are built per-county; 3, 4 and 5 are resampled up from 7.
STATE_H3_RESOLUTIONS: Sequence[int] = (3, 4, 5, 7, 8, 9, 10, 11)

#: Finest per-county h3 resolution coarser levels resample up from.
_BASE_H3_RESOLUTION = 7


def merge_counties(state_dir: Path) -> Path:
    """Concatenate every county's own `county.geoparquet` into one state-wide file.

    Args:
        state_dir: State folder containing a `counties/` subdirectory,
            one folder per county (as written by `county_pipeline
            .process_county`).

    Returns:
        Path to the written `county.geoparquet` (under `state_dir`
        directly, not inside `counties/`).
    """
    counties_dir = state_dir / "counties"
    frames = []
    for county_folder in sorted(counties_dir.iterdir()):
        county_path = county_folder / "county.geoparquet"
        if county_path.is_file():
            frames.append(gpd.read_parquet(county_path))
        else:
            logger.warning(f"No county.geoparquet for {county_folder.name} -- skipping in state rollup.")

    if not frames:
        raise ValueError(f"No county.geoparquet files found under {counties_dir}")

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    out_path = state_dir / "county.geoparquet"
    merged.to_parquet(out_path)
    return out_path


def merge_and_resample_h3(state_dir: Path, resolutions: Sequence[int] = STATE_H3_RESOLUTIONS) -> dict:
    """Build every state-level h3 resolution file from per-county h3 outputs.

    Args:
        state_dir: State folder containing a `counties/` subdirectory.
        resolutions: Target resolutions to write (see `STATE_H3_RESOLUTIONS`).
            Any resolution >= `_BASE_H3_RESOLUTION` (7) is a plain concat
            of the matching per-county files; anything coarser is
            resampled up from the concatenated res-7 table.

    Returns:
        Dict mapping resolution -> written file path (a resolution with
        no per-county data at all -- e.g. no counties had population --
        is simply absent).
    """
    counties_dir = state_dir / "counties"
    written = {}

    base_frames = []
    for county_folder in sorted(counties_dir.iterdir()):
        base_path = county_folder / f"h3_res{_BASE_H3_RESOLUTION}.geoparquet"
        if base_path.is_file():
            base_frames.append(pd.read_parquet(base_path))
    if not base_frames:
        logger.warning(f"No h3_res{_BASE_H3_RESOLUTION} files found under {counties_dir} -- skipping h3 rollup.")
        return written

    base = pd.concat(base_frames, ignore_index=True)

    for resolution in sorted(resolutions, reverse=True):
        if resolution == _BASE_H3_RESOLUTION:
            out = base
        elif resolution > _BASE_H3_RESOLUTION:
            # Finer than the base (dense tracts only) -- plain concat.
            frames = []
            for county_folder in sorted(counties_dir.iterdir()):
                p = county_folder / f"h3_res{resolution}.geoparquet"
                if p.is_file():
                    frames.append(pd.read_parquet(p))
            if not frames:
                continue
            out = pd.concat(frames, ignore_index=True)
        else:
            out = _resample_h3_up(base, resolution)

        if out.empty:
            continue
        from transitlos.neighbourhood import add_densities

        out = add_densities(out, "h3", "population")   # 500 m / 1 / 2 / 3 km neighbourhood densities, from this level's own cells
        path = state_dir / f"h3_res{resolution}.geoparquet"
        out.to_parquet(path)
        written[resolution] = str(path)

    return written


#: State-level census district levels, resampled from blockgroup via
#: centroid-in-polygon (per prompt.txt part 2). "local district" (city
#: council wards / county commissioner districts) is deliberately NOT
#: included -- confirmed no national dataset exists for it (see
#: [[usa_study_pipeline_progress]]).
STATE_DISTRICT_LEVELS = ("place", "congressional_district", "state_legislative_district", "school_district", "voting_district")


def _column_agg_map() -> dict:
    """Map every known census column name (bare and prefixed) to `"sum"`/`"mean"`.

    Pulled from each source's own `ColumnDef.agg` (the single place every
    part of pyCensus already records whether a column is additive or an
    intensive/relative quantity -- see `pycensus.schema.ColumnDef`), so
    this never has to independently guess/hardcode which of the ~90
    census columns this study carries are counts vs. rates.
    """
    from pycensus.countries.usa.acs5.schema import SCHEMA as ACS5_SCHEMA
    from pycensus.countries.usa.dhc.schema import SCHEMA as DHC_SCHEMA
    from pycensus.countries.usa.elections.schema import SCHEMA as ELECTIONS_SCHEMA
    from pycensus.countries.usa.lodes_wac.schema import SCHEMA as LODES_SCHEMA

    agg_map = {}
    for schema in (ACS5_SCHEMA, DHC_SCHEMA, LODES_SCHEMA, ELECTIONS_SCHEMA):
        for name, col in schema.columns.items():
            agg_map[name] = col.agg
            agg_map[schema.column_label(name)] = col.agg
    return agg_map


def _resample_by_centroid(
    blockgroup_gdf: gpd.GeoDataFrame, district_gdf: gpd.GeoDataFrame, id_col: str
) -> gpd.GeoDataFrame:
    """Resample blockgroup-level data onto a district polygon layer via centroid-in-polygon.

    Each blockgroup is assigned to the district polygon its centroid
    falls inside (per prompt.txt: "one blockgroup belongs to a place if
    the centroid of the blockgroup is inside the place polygon" --
    stated for "place" but applied identically to every district level).
    Count columns (population, jobs, ...) are summed; rate/mean columns
    (medians, shares, `level_of_service`) are population-weighted, per
    `_column_agg_map`.

    Args:
        blockgroup_gdf: State-wide blockgroup-level data (every county's
            `blockgroup.geoparquet` concatenated), with `level_of_service`
            and every joined census column.
        district_gdf: Target district polygons (from
            `pycensus.countries.usa.constants.GEOMETRY_FUNCS`), with the
            district's own id column.
        id_col: District id column name in `district_gdf`.

    Returns:
        `district_gdf`'s polygons with every resampled column.
    """
    agg_map = _column_agg_map()
    original_id_col = id_col
    bg = blockgroup_gdf.to_crs(4326)
    # `id_col` (typically "GEOID") collides with the blockgroup's OWN
    # "GEOID" column -- `gpd.sjoin` would silently auto-suffix one of
    # them (e.g. "GEOID_right") rather than raising, so rename the
    # district side to a distinct name up front instead of relying on
    # sjoin's suffixing convention.
    districts = district_gdf.to_crs(4326)[[id_col, "geometry"]].rename(columns={id_col: "_district_id"})

    centroids = gpd.GeoDataFrame(bg.drop(columns="geometry"), geometry=bg.geometry.centroid, crs=4326)
    joined = gpd.sjoin(centroids, districts, how="inner", predicate="within")
    id_col = "_district_id"

    # "level_of_service" isn't a real census schema column (it's added by
    # `county_pipeline`'s LOS join), so it's not in `agg_map` at all --
    # explicitly forced into the population-weighted-mean bucket rather
    # than falling through to `agg_map`'s "sum" default, which would
    # silently ADD every constituent blockgroup's LOS instead of
    # averaging it.
    numeric_cols = [c for c in bg.columns if c != "geometry" and pd.api.types.is_numeric_dtype(bg[c])]
    mean_cols = [
        c for c in numeric_cols if (agg_map.get(c) == "mean" or c == "level_of_service") and c != "population"
    ]
    count_cols = [c for c in numeric_cols if c not in mean_cols and agg_map.get(c, "sum") == "sum"]

    agg = {col: "sum" for col in count_cols}
    if mean_cols and "population" in joined.columns:
        for col in mean_cols:
            joined[f"_{col}_w"] = joined[col] * joined["population"]
        grouped = joined.groupby(id_col).agg({**agg, **{f"_{c}_w": "sum" for c in mean_cols}})
        pop_sum = grouped["population"].replace(0, pd.NA)
        for col in mean_cols:
            grouped[col] = (grouped[f"_{col}_w"] / pop_sum).fillna(0.0)
        grouped = grouped.drop(columns=[f"_{c}_w" for c in mean_cols])
    else:
        grouped = joined.groupby(id_col).agg(agg)

    result = districts.merge(grouped.reset_index(), on=id_col, how="inner")
    return result.rename(columns={"_district_id": original_id_col})


def _polygonal(geom):
    """Polygon / MultiPolygon part of a (possibly mixed) geometry, so tiling accepts it."""
    from shapely.geometry import MultiPolygon, Polygon

    if geom is None or geom.is_empty or isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    flat = [q for g in parts for q in (g.geoms if isinstance(g, MultiPolygon) else [g])]
    return MultiPolygon(flat) if flat else Polygon()


def build_voting_districts(state_dir: Path, state_fips: str, year: int = 2020) -> Optional[gpd.GeoDataFrame]:
    """Census voting districts (VTD, `year` vintage) carrying every blockgroup column, split by block population.

    Many VTDs are smaller than a block group, so a centroid join would lose them: each block group's counts are shared among the VTDs it
    overlaps in proportion to the block population in each part (block representative points decide the part), and its means /
    shares / `level_of_service` are population-weighted over those parts. Result has the `GEOID` column of the VTD (`GEOID20`).
    """
    import pygris

    try:
        vtd = pygris.voting_districts(state=state_fips, year=year, cache=True)
        id_src = next(c for c in vtd.columns if c.startswith("GEOID"))
        vtd = vtd.rename(columns={id_src: "GEOID"})[["GEOID", "geometry"]].to_crs(4326)
    except Exception as exc:  # noqa: BLE001 - the Census Bureau publishes no VTDs for California, Hawaii and Oregon
        from pycensus.cache import default_cache_dir
        from pycensus.countries.usa.elections_precinct.official import load_state_precincts

        logger.warning(f"No Census voting districts for state {state_fips} ({exc}); using the state's precinct polygons instead")
        prec, _ = load_state_precincts(state_fips, default_cache_dir())
        if prec is None:
            return None
        vtd = prec[["county_fips", "geometry"]].reset_index(drop=True)
        vtd["GEOID"] = vtd["county_fips"].astype(str) + "P" + vtd.index.astype(str).str.zfill(5)
        vtd = vtd[["GEOID", "geometry"]].to_crs(4326)
        vtd["geometry"] = vtd.geometry.apply(_polygonal)
        vtd = vtd[~vtd.geometry.is_empty]
    counties = sorted(p for p in (state_dir / "counties").iterdir() if p.is_dir())
    bgs, parts = [], []
    for cdir in counties:
        bgp, blp = cdir / "blockgroup.geoparquet", cdir / "block.geoparquet"
        if not (bgp.is_file() and blp.is_file()):
            continue
        bgs.append(gpd.read_parquet(bgp))
        blocks = gpd.read_parquet(blp)
        pts = gpd.GeoDataFrame({"bg": blocks["GEOID"].astype(str).str[:12].to_numpy(),
                                "pop": pd.to_numeric(blocks["population"], errors="coerce").fillna(0.0).to_numpy()},
                               geometry=blocks.to_crs(4326).geometry.representative_point(), crs=4326)
        j = gpd.sjoin(pts, vtd[["GEOID", "geometry"]], how="inner", predicate="within")
        parts.append(j.groupby(["bg", "GEOID"], as_index=False).agg(pop=("pop", "sum"), n=("pop", "size")))
    if not bgs or not parts:
        return None
    bg = pd.concat(bgs, ignore_index=True)
    w = pd.concat(parts, ignore_index=True).groupby(["bg", "GEOID"], as_index=False)[["pop", "n"]].sum()
    tot = w.groupby("bg")["pop"].transform("sum")
    cnt = w.groupby("bg")["n"].transform("sum")
    w["frac"] = (w["pop"] / tot.where(tot > 0)).fillna(w["n"] / cnt)
    agg_map = _column_agg_map()
    num = [c for c in bg.columns if c != "geometry" and pd.api.types.is_numeric_dtype(bg[c])]
    means = [c for c in num if (agg_map.get(c) == "mean" or c == "level_of_service") and c != "population"]
    counts = [c for c in num if c not in means]
    m = w[["bg", "GEOID", "frac"]].merge(bg.assign(bg=bg["GEOID"].astype(str)).drop(columns="GEOID")[["bg", *num]], on="bg", how="inner")
    pop_w = m["population"].fillna(0.0) * m["frac"]
    out = pd.DataFrame({c: m[c].fillna(0.0) * m["frac"] for c in counts})
    for c in means:
        out[f"_{c}"] = m[c].fillna(0.0) * pop_w
    out["GEOID"] = m["GEOID"].to_numpy()
    out["_w"] = pop_w.to_numpy()
    g = out.groupby("GEOID").sum()
    for c in means:
        g[c] = (g[f"_{c}"] / g["_w"].replace(0, np.nan)).fillna(0.0)
    g = g.drop(columns=[f"_{c}" for c in means] + ["_w"]).reset_index()
    res = vtd.merge(g, on="GEOID", how="inner")
    act = pd.to_numeric(res["population"], errors="coerce").fillna(0) > 0
    jobs = next((c for c in res.columns if "jobsTotal" in c), None)
    if jobs:
        act |= pd.to_numeric(res[jobs], errors="coerce").fillna(0) > 0
    return res[act]


def add_state_districts(
    state_dir: Path, state_fips: str, levels: Sequence[str] = STATE_DISTRICT_LEVELS
) -> dict:
    """Build every state-level census district file, resampled from blockgroup.

    Args:
        state_dir: State folder containing a `counties/` subdirectory.
        state_fips: State FIPS code (`pygris`'s geometry functions take
            this as `state=`).
        levels: Which district levels to build (see `STATE_DISTRICT_LEVELS`).

    Returns:
        Dict mapping level name -> written file path (a level with no
        pygris data for this state, or nothing surviving the
        population/jobs > 0 filter, is simply absent).
    """
    from pycensus.countries.usa.constants import GEOMETRY_FUNCS

    counties_dir = state_dir / "counties"
    bg_frames = [
        gpd.read_parquet(p)
        for p in (f / "blockgroup.geoparquet" for f in sorted(counties_dir.iterdir()))
        if p.is_file()
    ]
    if not bg_frames:
        logger.warning(f"No blockgroup.geoparquet files under {counties_dir} -- skipping state districts.")
        return {}
    blockgroup_gdf = gpd.GeoDataFrame(pd.concat(bg_frames, ignore_index=True), geometry="geometry", crs=bg_frames[0].crs)

    written = {}
    for level in levels:
        if level == "voting_district":
            vd = build_voting_districts(state_dir, state_fips)
            if vd is not None and not vd.empty:
                path = state_dir / "voting_district.geoparquet"
                vd.to_parquet(path)
                written[level] = str(path)
            continue
        if level == "state_legislative_district":
            import pygris

            chambers = []
            for house in ("upper", "lower"):
                try:
                    chamber_gdf = pygris.state_legislative_districts(state=state_fips, house=house, cache=True)
                except Exception as exc:  # pragma: no cover - Nebraska has no "lower" house
                    logger.warning(f"state_legislative_districts({house=}) failed for state {state_fips}: {exc}")
                    continue
                chamber_gdf["chamber"] = house
                chambers.append(chamber_gdf)
            if not chambers:
                continue
            district_gdf = gpd.GeoDataFrame(pd.concat(chambers, ignore_index=True), crs=chambers[0].crs)
            id_col = "GEOID"
        else:
            fetch = GEOMETRY_FUNCS[level]
            district_gdf = fetch(state=state_fips, cache=True)
            id_col = "GEOID"

        resampled = _resample_by_centroid(blockgroup_gdf, district_gdf, id_col)
        pop_col = "population" if "population" in resampled.columns else None
        jobs_col = next((c for c in resampled.columns if "jobsTotal" in c), None)
        has_activity = pd.Series(False, index=resampled.index)
        if pop_col:
            has_activity |= pd.to_numeric(resampled[pop_col], errors="coerce").fillna(0) > 0
        if jobs_col:
            has_activity |= pd.to_numeric(resampled[jobs_col], errors="coerce").fillna(0) > 0
        resampled = resampled[has_activity]
        if resampled.empty:
            continue

        path = state_dir / f"{level}.geoparquet"
        resampled.to_crs(4326).to_parquet(path)
        written[level] = str(path)

    return written


def add_state_native_elections(state_dir: Path, state_fips: str, state_abbr: str, cache_dir: Optional[str] = None) -> Optional[Path]:
    """Build `state.geoparquet`: state-NATIVE election results + resampled-up census.

    Per Miguel's 2026-09-29 direction: "add the native election levels as
    additional map levels that incorporate resampled data from all the
    other census columns." President/Senate are genuinely native at
    STATE level too (see `pycensus.countries.usa.elections.loader`'s
    module docstring -- a separate, independently-published MEDSL file
    each time, not a derived sum of the county file), so this is a
    distinct map level from `county.geoparquet`, carrying election
    columns that were fetched directly at this geography rather than
    disaggregated down.

    Args:
        state_dir: State folder containing a `counties/` subdirectory
            and (once `merge_counties` has run) its own `county.geoparquet`.
        state_fips: 2-digit state FIPS.
        state_abbr: 2-letter state abbreviation (`elections.load`'s
            per-state precinct/county source selector for some office/years).
        cache_dir: pyCensus cache dir.

    Returns:
        Path to the written `state.geoparquet` (a single-row
        GeoDataFrame), or `None` if no state-native election data was
        available for this state at all.
    """
    from pycensus.cache import default_cache_dir
    from pycensus.countries.usa import elections
    from pycensus.countries.usa.elections.loader import STATE_AVAILABLE
    from pycensus.countries.usa.geography import load_boundaries

    cache_dir = cache_dir or default_cache_dir()

    state_boundary = load_boundaries("state", state_fips, 2023)
    state_boundary = state_boundary[state_boundary["GEOID"] == state_fips].reset_index(drop=True)
    if state_boundary.empty:
        logger.warning(f"No state boundary found for FIPS {state_fips} -- skipping state.geoparquet.")
        return None

    result = state_boundary.copy()
    got_any_election_data = False
    for office, year in sorted(STATE_AVAILABLE):
        try:
            votes = elections.load(office, year, level="state", state_fips=state_fips, state_abbr=state_abbr, cache_dir=cache_dir)
        except Exception as exc:  # pragma: no cover - real network/source availability varies
            logger.warning(f"State-level elections {office}/{year} failed for {state_abbr}: {exc}")
            continue
        if votes.empty:
            continue

        # `elections.load` returns ONE ROW PER CANDIDATE (same shape as
        # `_add_elections`'s county-level input, confirmed live 2026-09-29
        # against real MA data: `votes[votes.GEOID==state_fips]` is 4-10
        # rows, not 1) -- must sum across candidates/parties, never take
        # `.iloc[0]` (that would silently treat one candidate's own vote
        # count as the state's total).
        rows = votes[votes["GEOID"] == state_fips]
        if rows.empty:
            continue
        prefix = f"{year}_{office}"
        total_votes = pd.to_numeric(rows["votes"], errors="coerce").fillna(0).sum()
        if total_votes <= 0:
            continue
        result[f"{prefix}_total_votes"] = total_votes

        # Party casing is inconsistent across MEDSL sources (lowercase
        # "democrat"/"republican" for pre-2024 constituency-returns,
        # uppercase "DEMOCRAT"/"REPUBLICAN" for 2024) -- same fix as
        # `_add_elections`.
        party_col = next((c for c in ("party", "party_simplified") if c in rows.columns), None)
        if party_col:
            for party in ("democrat", "republican"):
                party_votes = rows[rows[party_col].astype(str).str.lower() == party]["votes"]
                party_sum = pd.to_numeric(party_votes, errors="coerce").fillna(0).sum()
                result[f"{prefix}_{party}_share"] = party_sum / total_votes
            if f"{prefix}_democrat_share" in result and f"{prefix}_republican_share" in result:
                result[f"{prefix}_other_share"] = (1.0 - result[f"{prefix}_democrat_share"] - result[f"{prefix}_republican_share"]).clip(lower=0.0)
        got_any_election_data = True

    if not got_any_election_data:
        logger.warning(f"No state-native election data available for {state_abbr} ({state_fips}).")
        return None

    # Resample every county-level census column UP to this single state
    # row: counts summed (real totals), rate/mean columns population-weighted
    # (never a plain unweighted average of the counties).
    county_path = state_dir / "county.geoparquet"
    if county_path.is_file():
        county_gdf = gpd.read_parquet(county_path)
        agg_map = _column_agg_map()
        # Exclude election columns from this county->state resample: they
        # were just set directly above from the genuinely state-NATIVE
        # source (real MEDSL state-level totals, not a derived sum of the
        # county file -- see module docstring). Re-summing county.geoparquet's
        # OWN `_add_elections`-derived election columns here would silently
        # overwrite the state-native numbers with a re-aggregated county
        # figure -- confirmed live 2026-09-30: this bug produced
        # 2024_president_total_votes=3453931 (the COUNTY sum) instead of
        # the real state-native 3512930.
        numeric_cols = [
            c
            for c in county_gdf.columns
            if c not in ("geometry", "GEOID")
            and pd.api.types.is_numeric_dtype(county_gdf[c])
            and not _ELECTION_COL_RE.match(c)
        ]
        mean_cols = [c for c in numeric_cols if (agg_map.get(c) == "mean" or c == "level_of_service") and c != "population"]
        count_cols = [c for c in numeric_cols if c not in mean_cols and agg_map.get(c, "sum") == "sum"]

        for col in count_cols:
            result[col] = pd.to_numeric(county_gdf[col], errors="coerce").fillna(0).sum()
        if "population" in county_gdf.columns:
            pop = pd.to_numeric(county_gdf["population"], errors="coerce").fillna(0)
            pop_sum = pop.sum()
            for col in mean_cols:
                vals = pd.to_numeric(county_gdf[col], errors="coerce").fillna(0)
                result[col] = float((vals * pop).sum() / pop_sum) if pop_sum > 0 else 0.0
    else:
        logger.warning(f"No county.geoparquet under {state_dir} -- state.geoparquet will carry election data only.")

    path = state_dir / "state.geoparquet"
    result.to_crs(4326).to_parquet(path)
    return path


def _resample_h3_up(fine: pd.DataFrame, target_resolution: int) -> pd.DataFrame:
    """Aggregate a fine-resolution h3 table up to a coarser resolution.

    `population`/any `*_jobsTotal`-like count column: summed (a real
    total); share/intensive columns and `level_of_service`: population-weighted means over the
    all its children (every cell has a score), never a plain
    unweighted average (a near-zero-population cell shouldn't drag a
    coarser cell's real, population-experienced access score down/up as
    much as a heavily-populated one).
    """
    parent = fine["h3"].map(lambda cell: h3.cell_to_parent(cell, target_resolution)).to_numpy()
    codes, parents = pd.factorize(parent)
    try:
        agg_map = _column_agg_map()
    except Exception:
        agg_map = {}
    num_cols = [c for c in fine.columns if c not in ("h3", "level_of_service") and pd.api.types.is_numeric_dtype(fine[c])]
    is_mean = lambda c: agg_map.get(c) == "mean" or c.endswith(("_share", "_rate")) or any(  # noqa: E731 -- intensive columns
        m in c.lower() for m in ("median", "mean_", "_mean", "density", "_index", "_ratio"))
    sum_cols = [c for c in num_cols if not is_mean(c)]
    mean_cols = [c for c in num_cols if is_mean(c)]
    G = len(parents)

    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    first = sorted_codes[starts] if len(order) else np.zeros(0, dtype=np.int64)
    BLOCK = 16   # columns per pass: the nationwide res-8 table has ~8M rows x ~200 columns, whole-table copies ran out of memory

    def gsum(mat):   # column sums per parent: rows sorted once, then one reduceat per block of columns
        out = np.zeros((G, mat.shape[1]))
        if len(order):
            out[first] = np.add.reduceat(mat[order], starts, axis=0)
        return out

    cols = {"h3": parents}
    for b0 in range(0, len(sum_cols), BLOCK):
        names = sum_cols[b0:b0 + BLOCK]
        ssum = gsum(np.nan_to_num(fine[names].to_numpy(dtype=np.float64)))   # pandas' groupby-sum skips NaN (all-NaN group -> 0): same here
        for k, c in enumerate(names):
            cols[c] = ssum[:, k]
    if "population" in fine.columns:
        pop = pd.to_numeric(fine["population"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            if "level_of_service" in fine.columns:
                los = fine["level_of_service"].fillna(0.0).to_numpy(dtype=np.float64)
                jc = next((c for c in ("lodes_wac_jobsTotal", "jobs") if c in fine.columns), None)
                job = pd.to_numeric(fine[jc], errors="coerce").fillna(0).to_numpy(dtype=np.float64) if jc else np.zeros(len(fine))
                t = gsum(np.column_stack([los * pop, pop, los * job, job, los, np.ones(len(fine))]))
                # population-weighted; where nobody lives jobs-weighted; where neither, the plain mean of the children (the county rule)
                cols["level_of_service"] = np.where(t[:, 1] > 0, t[:, 0] / np.where(t[:, 1] > 0, t[:, 1], 1),
                                                    np.where(t[:, 3] > 0, t[:, 2] / np.where(t[:, 3] > 0, t[:, 3], 1), t[:, 4] / np.maximum(t[:, 5], 1)))
            for b0 in range(0, len(mean_cols), BLOCK):
                names = mean_cols[b0:b0 + BLOCK]
                v = fine[names].to_numpy(dtype=np.float64)
                ok = np.isfinite(v)
                tot = gsum(np.concatenate([np.where(ok, v * pop[:, None], 0.0), np.where(ok, pop[:, None], 0.0)], axis=1))
                m = len(names)
                for k, c in enumerate(names):
                    w, p_ = tot[:, k], tot[:, m + k]
                    cols[c] = np.where(p_ > 0, w / np.where(p_ > 0, p_, 1), np.nan)
    grouped = pd.DataFrame(cols)
    return grouped
