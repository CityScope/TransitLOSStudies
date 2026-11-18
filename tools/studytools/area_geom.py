from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

_NE = Path(__file__).resolve().parents[2] / "cache" / "natural_earth" / "ne_10m_admin_0_countries.shp"
_RATIO = 0.35
_cut = {}


def _single(g):
    if isinstance(g, Polygon):
        return g
    parts = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
    return max(parts, key=lambda p: p.area) if parts else g


def _cutout():
    if "g" not in _cut:
        ne = gpd.read_file(_NE)
        _cut["g"] = ne[ne["NAME"].eq("Palestine")].to_crs(4326).union_all().buffer(0)
    return _cut["g"]


def _join(geom, city: str = ""):
    """The area as one polygon: separate parts are joined by a moderate concave hull (never more concave than needed to connect them)."""
    g = shapely.make_valid(geom)
    if city == "beerseba":
        g = g.difference(_cutout())
    parts = list(getattr(g, "geoms", [g]))
    if city == "beerseba" and len(parts) > 1:
        hull = shapely.concave_hull(g.simplify(0.0005), ratio=0.5, allow_holes=False)
        g = unary_union([g, hull]).buffer(0.012).buffer(-0.012).difference(_cutout())
        return _single(_single(g).simplify(0.0015).difference(_cutout()))
    if len(parts) > 1:
        tol = 0.0005
        hull = shapely.concave_hull(g.simplify(tol), ratio=_RATIO, allow_holes=False)
        g = unary_union([g, hull])
        if city == "beerseba":
            g = g.difference(_cutout())
        if len(getattr(g, "geoms", [g])) > 1:
            g = g.buffer(0.02).buffer(-0.02)
            if city == "beerseba":
                g = g.difference(_cutout())
        g = _single(g)
    return g


def _solid(g):
    return Polygon(g.exterior) if isinstance(g, Polygon) else g


def one_piece(geom, city: str = ""):
    """One polygon without holes or separate parts."""
    g = _single(_join(geom, city))
    if isinstance(g, Polygon) and g.interiors:
        g = _solid(g)
        if city == "beerseba":
            g = _single(g.difference(_cutout()))
    return g
