"""Level of service of h3 cells with and without streets: the city science network's rule, shared by the US study.

  * LOS of a cell = the best (Max) LOS among street edges within reach of its centroid, falling back to the nearest street when none is
    within `margin_m` (`geohierarchy.edges_to_h3_by_distance`, the same call and parameters as `city_science_network/code/h3_population.py`);
  * a cell that touches a street but that no scored edge reached has LOS 0 (a cell with no transit access, not an unknown one);
  * a cell with no street within reach cannot exist in the grid: its population is moved onto the NEAREST street-touching cell (by cell
    centroid distance), but only within `MAX_REASSIGN_DIST_M`; further away it is dropped. Nothing is ever left "unscored".
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

logger = logging.getLogger(__name__)
MARGIN_M = 10.0
FALLBACK_RADIUS_MULTIPLIER = 5.0
MAX_REASSIGN_DIST_M = 300.0   # = city_science_network/code/h3_population.py `max_reassign_dist_m` (its docstring says 500, the code uses 300)
EDGE_CHUNK = 500_000


def sample_cell_los(scored_edges_wkb, street_geoms: Optional[gpd.GeoSeries], cells: Sequence[str], resolution: int, max_dist_m: Optional[float] = None) -> pd.Series:
    """`Series(index=cell id)` of LOS for the cells that have a street within reach (others are absent = roadless).

    `scored_edges_wkb`: polars table (EPSG:4326 WKB `geometry_wkb`/`geometry` + `level_of_service`) of the edges with LOS > 0;
    `street_geoms`: every street line of the county (EPSG:4326, scored or not) -- they count as LOS 0 so a cell touching only
    unreached streets is street-touching with LOS 0. `None`: street presence unknown, only scored edges count.
    """
    from geohierarchy import Max, edges_to_h3_by_distance

    parts = []
    if scored_edges_wkb is not None and scored_edges_wkb.height:
        geom_col = "geometry" if "geometry" in scored_edges_wkb.columns else "geometry_wkb"
        parts.append(gpd.GeoDataFrame({"level_of_service": scored_edges_wkb["level_of_service"].to_numpy().astype(float)},
                                      geometry=shapely.from_wkb(scored_edges_wkb[geom_col].to_numpy()), crs=4326))
    if street_geoms is not None and len(street_geoms):
        parts.append(gpd.GeoDataFrame({"level_of_service": np.zeros(len(street_geoms))}, geometry=street_geoms.values, crs=4326))
    if not parts:
        return pd.Series(dtype=float)
    edges = pd.concat(parts, ignore_index=True)
    import h3 as _h3

    fallback = FALLBACK_RADIUS_MULTIPLIER if max_dist_m is None else max(FALLBACK_RADIUS_MULTIPLIER, max_dist_m / _h3.average_hexagon_edge_length(resolution, unit="m") + 1.0)
    best: dict = {}
    cells = list(cells)
    for start in range(0, len(edges), EDGE_CHUNK):   # Max() is chunk-safe: the max of the chunk maxima is the overall max
        chunk = edges.iloc[start:start + EDGE_CHUNK]
        res = edges_to_h3_by_distance(chunk, cells, "level_of_service", resolution, margin_m=MARGIN_M, agg=Max(),
                                      fallback_radius_multiplier=fallback)
        for cell, v in zip(res["h3_cell"].to_list(), res["level_of_service"].to_list()):
            if v is not None and (cell not in best or v > best[cell]):
                best[cell] = float(v)
    return pd.Series(best, dtype=float)


def move_roadless_population(grid: pd.DataFrame, touching: pd.Series, columns: Sequence[str] = ("population",), max_dist_m: float = MAX_REASSIGN_DIST_M,
                             resolution: Optional[int] = None) -> pd.DataFrame:
    """Drop the cells without a street (`touching` False) after moving their `columns` onto the nearest street-touching cell.

    `grid` needs `h3` and a `geometry` (cells' own polygons, EPSG:4326). Cells further than `MAX_REASSIGN_DIST_M` from any
    street-touching cell lose their people (logged), exactly like the city studies.
    """
    from scipy.spatial import cKDTree

    touching = touching.reindex(grid.index).fillna(False).astype(bool)
    if touching.all():
        return grid
    cent = grid.geometry.centroid
    lat0 = float(np.deg2rad(cent.y.mean()))
    xy = np.column_stack([np.deg2rad(cent.x.to_numpy()) * np.cos(lat0), np.deg2rad(cent.y.to_numpy())]) * 6_371_000.0
    keep_idx = np.flatnonzero(touching.to_numpy())
    if len(keep_idx) == 0:
        logger.warning("no street-touching cell: the county's roadless population is dropped")
        return grid.iloc[0:0]
    tree = cKDTree(xy[keep_idx])
    drop_idx = np.flatnonzero(~touching.to_numpy())
    dist, nearest = tree.query(xy[drop_idx])
    out = grid.copy()
    for col in columns:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy().copy()
        total, roadless = float(vals.sum()), float(vals[drop_idx].sum())
        near = dist <= max_dist_m
        np.add.at(vals, keep_idx[nearest[near]], vals[drop_idx[near]])
        dropped = float(vals[drop_idx[~near]].sum())
        log_roadless(f"census blocks (res {resolution})" if resolution else "census blocks", col, total, roadless, roadless - dropped, dropped, max_dist_m)
        out[col] = vals
    return out[touching.to_numpy()]


from .study_params import GRID as _GRID

ROADLESS_WARN_SHARE = _GRID.rescale_warn_share   # `study_params.GRID`


def log_roadless(source: str, what: str, total: float, roadless: float, moved: float, dropped: float, max_reassign_dist_m: float, rescaled_to=None) -> None:
    """One log line per source and column: how much is not on a kept cell (off-road or in an uninhabited cell), the rescaling factor to
    the source's total (applied when `rescaled_to` is given), and a WARNING when more than `ROADLESS_WARN_SHARE` is affected (same format
    as the city studies)."""
    if total <= 0 or roadless <= 0:
        return
    share = roadless / total
    if rescaled_to is not None:
        kept = total - roadless
        msg = (f"[roadless] {source} {what}: {roadless:,.0f} of {total:,.0f} ({share:.1%}) not within {max_reassign_dist_m:.0f} m of a street or in "
               f"uninhabited cells -- rescaled into the kept cells, factor {(rescaled_to / kept if kept > 0 else float('nan')):.4f} (applied to every cell)")
    else:
        kept = total - dropped
        msg = (f"[roadless] {source} {what}: {roadless:,.0f} of {total:,.0f} ({share:.1%}) not near a road -- {moved:,.0f} moved to the nearest "
               f"street cell within {max_reassign_dist_m:.0f} m, {dropped:,.0f} dropped; rescaling factor to the source total "
               f"{(total / kept if kept > 0 else float('nan')):.4f} (not applied)")
    if share > ROADLESS_WARN_SHARE:
        logger.warning(msg + f" -- over {ROADLESS_WARN_SHARE:.0%} affected")
    else:
        logger.info(msg)


def coarse_cell_touching(grid: pd.DataFrame, street_geoms: Optional[gpd.GeoSeries], resolution: int) -> pd.Series:
    """Bool Series: does each cell of a COARSE grid (res 7/8, whole county) touch a street (same reach rule as res 11)?

    `street_geoms` None (street presence unknown): every cell counts as touching."""
    if street_geoms is None or not len(street_geoms):
        return pd.Series(True, index=grid.index)
    sampled = sample_cell_los(None, street_geoms, grid["h3"].tolist(), resolution)
    return grid["h3"].isin(sampled.index)


def coarse_reassign_dist_m(resolution: int) -> float:
    """Reassignment cap for coarse cells: the city rule's 500 m, but never less than 3 cell edges (a res 7 edge is ~1.2 km)."""
    import h3

    return max(MAX_REASSIGN_DIST_M, 3.0 * h3.average_hexagon_edge_length(resolution, unit="m"))


def los_from_children(grid: pd.DataFrame, fine: Optional[pd.DataFrame], resolution: int) -> pd.Series:
    """LOS of a coarser cell from its finer children, the city science rule (`_resample_h3`): population-weighted mean of the children's LOS.

    The finest table (res 11) only covers the buffered scored streets + dense tracts, so the people of the coarse cell that no child holds
    are street-touching but unscored (LOS 0, exactly what the city grid would hold as children with LOS 0): the denominator is therefore
    max(coarse cell population, children population). A cell whose population is 0 takes the unweighted mean of its children; a cell
    without any child has LOS 0."""
    import h3

    if fine is None or fine.empty:
        return pd.Series(0.0, index=grid.index)
    k = fine[["h3", "level_of_service", "population"]].copy()
    k["_p"] = [h3.cell_to_parent(c, resolution) for c in k["h3"]]
    los = pd.to_numeric(k["level_of_service"], errors="coerce").fillna(0.0)
    pop = pd.to_numeric(k["population"], errors="coerce").fillna(0.0)
    g = pd.DataFrame({"_p": k["_p"], "w": los * pop, "pop": pop, "los": los, "n": 1.0}).groupby("_p").sum()
    gp = grid["h3"].map(g["pop"]).fillna(0.0)
    gw = grid["h3"].map(g["w"]).fillna(0.0)
    cell_pop = pd.to_numeric(grid["population"], errors="coerce").fillna(0.0) if "population" in grid.columns else gp
    denom = np.maximum(cell_pop.to_numpy(), gp.to_numpy())
    weighted = pd.Series(np.where(denom > 0, gw.to_numpy() / np.where(denom > 0, denom, 1.0), np.nan), index=grid.index)
    plain = grid["h3"].map(g["los"] / g["n"])
    return weighted.fillna(plain).fillna(0.0)
