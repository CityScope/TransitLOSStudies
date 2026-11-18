"""Replace the county-broadcast 2024 presidential columns by results at VOTING-DISTRICT level, resampled to every other level.

Precinct results (official state data where a loader exists, else the NYT precinct map) are assigned to the Census voting districts
(`voting_district.geoparquet`, the native unit for votes). Per county, every census block's population-weighted share of the voting
district containing its representative point gives block votes; block groups and tracts sum their blocks (GEOID prefixes), H3 res 8 sums
the blocks whose point falls in the cell, coarser H3 resolutions sum res 8, finer ones split their res 8 parent by population; places /
school districts / legislative districts are filled by `allocate_to_polygons`. County and state rows keep their native MEDSL result.
Units without voting-district coverage keep the county value.

    python add_precinct_elections.py [State ...]      # default: every state folder in data/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from pycensus.cache import default_cache_dir
from pycensus.census.countries.usa.elections_precinct import nyt2024
from pycensus.census.countries.usa.elections_precinct.official import REGISTRY, load_state_precincts

import os
DATA = Path(os.environ.get("US_DATA") or Path(__file__).parent / "data")
COLS = ("2024_president_eligible_voters", "2024_president_total_votes", "2024_president_democrat_share", "2024_president_republican_share", "2024_president_other_share", "2024_president_turnout")
POLY_LEVELS = ("county", "state", "place", "school_district", "congressional_district", "state_legislative_district")
MARK = "vtd_native_elections_2024.done"
logger = logging.getLogger("precinct_elections")


def _write(df, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.replace(path)


def _set_votes(df, dem, rep, tot, vap=None):
    """Write total votes, the three party shares and turnout (all over the voting-age population) where the votes are known (> 0); other rows keep their value."""
    tot = np.asarray(tot, dtype=float)
    ok = np.isfinite(tot) & (tot > 0)
    if not ok.any() or (vap is None and not {"population", "under18Population"} <= set(df.columns)):
        return df
    if vap is None:
        vap = pd.to_numeric(df["population"], errors="coerce").to_numpy() - pd.to_numeric(df["under18Population"], errors="coerce").to_numpy()
    vap = np.asarray(vap, dtype=float)
    den = np.where(ok & (vap > 0), vap, np.nan)
    dem, rep = np.asarray(dem, dtype=float), np.asarray(rep, dtype=float)
    new = (np.where(ok & (vap > 0), vap, np.nan), tot, dem / den, rep / den, np.clip(tot - dem - rep, 0.0, None) / den, tot / den)
    for col, vals in zip(COLS, new):
        old = pd.to_numeric(df[col], errors="coerce").to_numpy() if col in df.columns else np.full(len(df), np.nan)
        keep = ok if col.endswith("total_votes") else ok & np.isfinite(vals)   # eligible_voters / ratios need a positive voting-age population
        df[col] = np.where(keep, vals, old)
    return df


def _county_blocks(county_dir: Path, precincts: gpd.GeoDataFrame):
    """Block votes (DataFrame indexed like the block file's GEOID) for one county, or None."""
    path = county_dir / "block.geoparquet"
    if not path.is_file():
        return None
    blocks = gpd.read_parquet(path).reset_index(drop=True)   # the file keeps gaps in its index: geometry must stay aligned with the arrays below
    prec = precincts[precincts["county_fips"] == county_dir.name].reset_index(drop=True)
    if prec.empty and not precincts["county_fips"].isin({p.name for p in county_dir.parent.iterdir()}).any():
        # the source codes counties differently (Connecticut: planning regions vs the legacy counties of the block files): take the polygons by location
        minx, miny, maxx, maxy = blocks.total_bounds
        prec = precincts.cx[minx:maxx, miny:maxy].reset_index(drop=True)
    if prec.empty or "population" not in blocks.columns:
        return None
    under = pd.to_numeric(blocks["under18Population"], errors="coerce").fillna(0.0).to_numpy() if "under18Population" in blocks.columns else 0.0
    pts = gpd.GeoDataFrame({"GEOID": blocks["GEOID"].to_numpy(), "pop": pd.to_numeric(blocks["population"], errors="coerce").fillna(0.0).to_numpy(),
                            "vap": pd.to_numeric(blocks["population"], errors="coerce").fillna(0.0).to_numpy() - under},
                           geometry=blocks.geometry.representative_point(), crs=blocks.crs)
    j = gpd.sjoin(pts, prec[["votes_dem", "votes_rep", "votes_total", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    pid = j["index_right"]
    pop = j["pop"]
    group_pop = pop.groupby(pid).transform("sum")
    group_n = pid.groupby(pid).transform("size")
    frac = np.where(group_pop > 0, pop / group_pop.replace(0, np.nan), 1.0 / group_n)
    frac = pd.Series(frac, index=j.index).where(pid.notna())
    out = pd.DataFrame({"GEOID": j["GEOID"].to_numpy(), "vap": j["vap"].to_numpy(), "lat": pts.geometry.y.to_numpy(), "lon": pts.geometry.x.to_numpy()}, index=j.index)
    for c in ("votes_dem", "votes_rep", "votes_total"):
        out[c] = (j[c] * frac).to_numpy()
    out = out.dropna(subset=["votes_total"])
    # districts holding no block point (small or all-unpopulated by point rule): share their votes among the blocks they overlap, by population x overlap
    miss = sorted(set(range(len(prec))) - set(pid.dropna().astype(int)))
    if miss:
        crs = 5070
        bl = blocks[["GEOID", "geometry"]].assign(_pop=pd.to_numeric(blocks["population"], errors="coerce").fillna(0.0).to_numpy(), _vap=pts["vap"].to_numpy(),
                                                    _lat=pts.geometry.y.to_numpy(), _lon=pts.geometry.x.to_numpy()).to_crs(crs)
        bl["geometry"] = bl.geometry.make_valid()
        bl["_area"] = bl.geometry.area.replace(0, np.nan)
        pr = prec.loc[miss, ["votes_dem", "votes_rep", "votes_total", "geometry"]].assign(_p=miss).to_crs(crs)
        pr["geometry"] = pr.geometry.make_valid()
        pieces = gpd.overlay(bl, pr, how="intersection", keep_geom_type=False)
        if not pieces.empty:
            pieces["_w"] = pieces["_pop"] * (pieces.geometry.area / pieces["_area"]).fillna(0.0)
            tot = pieces.groupby("_p")["_w"].transform("sum")
            pieces = pieces[tot > 0]
            fr = pieces["_w"] / pieces.groupby("_p")["_w"].transform("sum")
            extra = pd.DataFrame({"GEOID": pieces["GEOID"].to_numpy(), "vap": (pieces["_vap"] * (fr > 0)).to_numpy() * 0.0, "lat": pieces["_lat"].to_numpy(), "lon": pieces["_lon"].to_numpy()})
            for c in ("votes_dem", "votes_rep", "votes_total"):
                extra[c] = (pieces[c] * fr).to_numpy()
            out = pd.concat([out, extra], ignore_index=True)
    return out


def _patch_county_levels(county_dir: Path, blocks: pd.DataFrame) -> None:
    sums = blocks.set_index("GEOID")[["votes_dem", "votes_rep", "votes_total"]]
    for level, n in (("block", 15), ("blockgroup", 12), ("tract", 11)):
        path = county_dir / f"{level}.geoparquet"
        if not path.is_file():
            continue
        g = gpd.read_parquet(path)
        agg = sums.groupby(sums.index.str[:n]).sum()
        v = agg.reindex(g["GEOID"].astype(str).to_numpy())
        g = _set_votes(g, v["votes_dem"].to_numpy(), v["votes_rep"].to_numpy(), v["votes_total"].to_numpy())
        _write(g, path)


def _res8_table(all_blocks: list) -> pd.DataFrame:
    b = pd.concat(all_blocks)
    b["cell"] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(b["lat"].to_numpy(), b["lon"].to_numpy())]
    return b.groupby("cell")[["votes_dem", "votes_rep", "votes_total", "vap"]].sum()


def _patch_h3(path: Path, res8: pd.DataFrame) -> None:
    df = pd.read_parquet(path)
    if df.empty or "h3" not in df.columns:
        return
    cells = df["h3"].astype(str).to_numpy()
    r = h3.get_resolution(cells[0])
    if r <= 8:
        if r == 8:
            v = res8.reindex(cells)
        else:
            parent = pd.Series([h3.cell_to_parent(c, r) for c in res8.index], index=res8.index)
            v = res8.groupby(parent.to_numpy()).sum().reindex(cells)
        vals = [v[c].to_numpy() for c in ("votes_dem", "votes_rep", "votes_total", "vap")]
    else:
        parent = np.array([h3.cell_to_parent(c, 8) for c in cells])
        pop = pd.to_numeric(df["population"], errors="coerce").fillna(0.0).to_numpy()
        ps = pd.Series(pop).groupby(parent).transform("sum").to_numpy()
        n = pd.Series(pop).groupby(parent).transform("size").to_numpy()
        frac = np.where(ps > 0, pop / np.where(ps > 0, ps, 1.0), 1.0 / n)
        v = res8.reindex(parent)
        vals = [v[c].to_numpy() * frac for c in ("votes_dem", "votes_rep", "votes_total", "vap")]
    for i, c in enumerate(("votes_dem", "votes_rep", "votes_total")):
        # cells missing from the file (water, unpopulated) would take their votes with them: one factor keeps the file's total equal to the precincts'
        got, want = float(np.nansum(vals[i])), float(res8[c].sum())
        if got > 0 and want > 0:
            vals[i] = vals[i] * (want / got)
    for col in COLS:   # drop the earlier (county-level / NYT-derived) values: cells the precinct votes do not reach stay empty
        if col in df.columns:
            df[col] = np.nan
    df = _set_votes(df, *vals)
    _write(df, path)


def _vtd_votes(state_dir: Path, precincts):
    """Native voting-district votes: precinct results assigned to the state's Census voting districts (written back to voting_district.geoparquet)."""
    path = state_dir / "voting_district.geoparquet"
    if not path.is_file():
        return None
    vtd = gpd.read_parquet(path)
    v = nyt2024.allocate_to_polygons(vtd, precincts, pop_col="population")
    if v is None:
        return None
    # votes allocated to dropped (zero-population) districts or outside coverage are not lost: rescale so each county keeps its precinct total
    cty = vtd["GEOID"].astype(str).str[:5].to_numpy()
    for col in ("votes_dem", "votes_rep", "votes_total"):
        want = precincts.groupby("county_fips")[col].sum()
        have = v[col].groupby(cty).sum()
        f = (want / have.where(have > 0)).reindex(cty).fillna(1.0).to_numpy()
        v[col] = v[col].to_numpy() * np.clip(f, 1.0, 3.0)
    vtd = _set_votes(vtd, v["votes_dem"].to_numpy(), v["votes_rep"].to_numpy(), v["votes_total"].to_numpy())
    _write(vtd, path)
    out = vtd[["GEOID", "geometry"]].copy()
    out["county_fips"] = out["GEOID"].astype(str).str[:5]
    out["votes_dem"], out["votes_rep"], out["votes_total"] = v["votes_dem"].to_numpy(), v["votes_rep"].to_numpy(), v["votes_total"].to_numpy()
    return out[out["votes_total"].fillna(0) > 0].reset_index(drop=True)


def _patch_vtd(state_dir: Path, collected: list) -> None:
    """Census voting districts as a resampled level: sum of the votes of the blocks whose point lies in each district (same rule as tracts)."""
    path = state_dir / "voting_district.geoparquet"
    if not path.is_file():
        return
    vtd = gpd.read_parquet(path)
    b = pd.concat(collected)
    pts = gpd.GeoDataFrame(b[["votes_dem", "votes_rep", "votes_total", "vap"]].reset_index(drop=True),
                           geometry=gpd.points_from_xy(b["lon"].to_numpy(), b["lat"].to_numpy()), crs=4326)
    j = gpd.sjoin(pts, vtd[["geometry"]].reset_index(drop=True), how="inner", predicate="within")
    j = j[~j.index.duplicated(keep="first")]   # overlapping voting districts must not count a block twice
    s = j.groupby("index_right")[["votes_dem", "votes_rep", "votes_total", "vap"]].sum().reindex(range(len(vtd)))
    # votes of blocks that fall in no (or in a dropped, zero-population) voting district must not vanish: each county keeps its block total
    cty_v = vtd["GEOID"].astype(str).str[:5].to_numpy()
    want = b.assign(_c=b["GEOID"].astype(str).str[:5]).groupby("_c")[["votes_dem", "votes_rep", "votes_total"]].sum()
    for col in ("votes_dem", "votes_rep", "votes_total"):
        have = s[col].groupby(cty_v).sum()
        f = (want[col] / have.where(have > 0)).reindex(cty_v).fillna(1.0).to_numpy()
        s[col] = s[col].to_numpy() * np.clip(f, 1.0, 4.0)
    for col in ("votes_dem", "votes_rep", "votes_total"):   # whatever the districts could not hold (no district covers the block) is shared out, so the layer sums to the blocks'
        have_all, want_all = float(np.nansum(s[col].to_numpy())), float(b[col].sum())
        if have_all > 0 and want_all > 0:
            s[col] = s[col].to_numpy() * (want_all / have_all)
    for col in COLS:   # rows no vote reaches must not keep an older (county-derived) value
        if col in vtd.columns:
            vtd[col] = np.nan
    vtd = _set_votes(vtd, s["votes_dem"].to_numpy(), s["votes_rep"].to_numpy(), s["votes_total"].to_numpy(), s["vap"].to_numpy())
    _write(vtd, path)


def process_state(state_dir: Path) -> str:
    """The official election districts are the native unit for votes: they go to blocks and every other level, Census voting districts included."""
    if (state_dir / MARK).exists():
        return "already done"
    counties = sorted(p for p in (state_dir / "counties").glob("*") if p.is_dir()) if (state_dir / "counties").is_dir() else []
    if not counties:
        return "no counties"
    fips = counties[0].name[:2]
    if not (state_dir / "voting_district.geoparquet").is_file():
        from code.state_pipeline import build_voting_districts
        g = build_voting_districts(state_dir, fips)
        if g is not None and not g.empty:
            g.to_parquet(state_dir / "voting_district.geoparquet")
    raw, attribution = load_state_precincts(fips, default_cache_dir())
    a = str(attribution)
    if fips in REGISTRY:
        source = "official"
    elif "Oklahoma statewide voting precincts" in a:
        source = "official precinct polygons + MEDSL votes (unofficial)"
    elif "spread over that district's precincts" in a:
        source = "unofficial (MEDSL precincts + district-level ballots on Census voting districts, calibrated to FEC)"
    elif "ESTIMATED" in a:
        source = "unofficial (NYT polygons and votes; uncovered precincts ESTIMATED from VEST 2020, calibrated to FEC)"
    elif "county totals scaled" in a:
        source = "unofficial (NYT polygons and votes, county totals validated against MEDSL)"
    elif "Precinct polygons and votes: The New York Times" in a:
        source = "unofficial (NYT polygons and votes, calibrated to FEC state totals)"
    else:
        source = "unofficial (MEDSL + NYT, cross-validated)"
    if raw is None:
        (state_dir / MARK).write_text("no precinct data\n")
        return "no precinct data (county results kept)"
    # the official election districts are the native geometry: they go straight to blocks; the Census voting-district layer is a resampled level
    precincts = raw[raw["votes_total"].fillna(0) > 0].reset_index(drop=True)
    if (state_dir / "county.geoparquet").is_file():
        # county codes of the source can differ from the study's county folders (Connecticut planning regions, Alaska boroughs / census areas): use the county holding the polygon's centre
        cty = gpd.read_parquet(state_dir / "county.geoparquet")[["GEOID", "geometry"]].to_crs(precincts.crs)
        hit = gpd.sjoin(gpd.GeoDataFrame(geometry=precincts.geometry.representative_point(), crs=precincts.crs), cty, how="left", predicate="within")
        hit = hit[~hit.index.duplicated(keep="first")]
        precincts["county_fips"] = hit["GEOID"].reindex(precincts.index).fillna(precincts["county_fips"]).astype(str).to_numpy()
    collected = []
    for cdir in counties:
        blocks = _county_blocks(cdir, precincts)
        if blocks is None or blocks.empty:
            continue
        _patch_county_levels(cdir, blocks)
        collected.append(blocks)
    if not collected:
        (state_dir / MARK).write_text("no matching precincts\n")
        return "no precinct matches"
    want_total = {c: float(precincts[c].sum()) for c in ("votes_dem", "votes_rep", "votes_total")}
    for c, w in want_total.items():   # votes of polygons that hold no block point (water, empty land) must not vanish
        have = float(sum(b[c].sum() for b in collected))
        if have > 0 and w > 0 and abs(have / w - 1) > 1e-9:
            for b in collected:
                b[c] = b[c] * (w / have)
    _patch_vtd(state_dir, collected)
    res8 = _res8_table(collected)
    for cdir in counties:
        for path in sorted(cdir.glob("h3_res*.geoparquet")):
            _patch_h3(path, res8)
    for path in sorted(state_dir.glob("h3_res*.geoparquet")):
        _patch_h3(path, res8)
    for level in POLY_LEVELS:
        path = state_dir / f"{level}.geoparquet"
        if not path.is_file():
            continue
        g = gpd.read_parquet(path)
        v = nyt2024.allocate_to_polygons(g, precincts, pop_col="population")
        if v is not None:
            for col in COLS:   # areas no precinct reaches must not keep an older county-derived value
                if col in g.columns:
                    g[col] = np.nan
            g = _set_votes(g, v["votes_dem"].to_numpy(), v["votes_rep"].to_numpy(), v["votes_total"].to_numpy())
            _write(g, path)
    (state_dir / MARK).write_text(f"ok {source}\n{attribution}\n")
    return f"ok {source} ({len(raw)} official districts)"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    wanted = sys.argv[1:] or sorted(p.name for p in DATA.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for name in wanted:
        d = DATA / name
        if d.is_dir():
            try:
                print(f"[precinct] {name}: {process_state(d)}", flush=True)
            except Exception as exc:  # noqa: BLE001 - one failing state must not stop the others
                print(f"[precinct] {name}: FAILED {type(exc).__name__}: {str(exc)[:150]}", flush=True)
