"""Write the metro (aoi.gpkg) and core (core.gpkg) boundaries of every city from the UN Degree of Urbanisation areas (pycensus.worldwide.un_metro_areas).

    metro = functional urban area (urban centre + commuting zone), core = urban centre, both without holes.
    Exceptions: andorra (core = Andorra la Vella municipality, metro = the whole country: files kept), gipuzkoa (metro = the province: file kept,
    core = the UN urban centre of Donostia / San Sebastian).
The previous files are kept as aoi_prev.gpkg. Run with --dry to only print the comparison with the current area.
"""
import shutil, sys
from pathlib import Path
import geopandas as gpd, pandas as pd

ROOT = Path(__file__).resolve().parent
from pycensus.worldwide import un_metro_areas as un

CACHE = str(ROOT.parent / "cache" / "pycensus")
POINTS = {"toronto": (-79.38, 43.65), "boston": (-71.06, 42.36), "san_francisco": (-122.42, 37.77), "hamburg": (9.99, 53.55), "taipei": (121.56, 25.04),
          "gipuzkoa": (-1.98, 43.32), "beerseba": (34.79, 31.25), "concepcion": (-73.05, -36.82), "guadalajara": (-103.35, 20.67), "shanghai": (121.47, 31.23)}
KEEP_METRO = {"gipuzkoa"}   # province polygon already in aoi.gpkg
dry = "--dry" in sys.argv
NE = ROOT.parent / "cache" / "natural_earth" / "ne_10m_admin_0_countries.shp"


def _excluded_territory():
    if not NE.is_file():
        import io, zipfile, requests
        NE.parent.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(requests.get("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip", timeout=300).content)).extractall(NE.parent)
    n = gpd.read_file(NE)
    return n[n["ADM0_A3"] == "PSX"].to_crs(4326).union_all()

for city, pt in POINTS.items():
    cur = gpd.read_file(ROOT / city / "aoi.gpkg").to_crs(4326)
    metro = un.find(point=pt, kind="functional_urban_area", cache_dir=CACHE).iloc[0:1]
    core = un.find(point=pt, kind="urban_centre", cache_dir=CACHE).iloc[0:1]
    m_new = metro.geometry.iloc[0]
    eq = "EPSG:6933"
    cu = gpd.GeoSeries([cur.union_all()], crs=4326).to_crs(eq).iloc[0]
    nw = gpd.GeoSeries([m_new], crs=4326).to_crs(eq).iloc[0]
    outside = nw.difference(cu).area / nw.area
    print(f"{city:14s} new metro {metro['name'].iloc[0]!r} {nw.area/1e6:7.0f} km2 | current {cu.area/1e6:7.0f} km2 | new area outside current: {outside:5.1%}")
    if city == "beerseba":
        pal = _excluded_territory()
        metro = metro.assign(geometry=metro.geometry.difference(pal))
        core = core.assign(geometry=core.geometry.difference(pal))
    if dry:
        continue
    prev = ROOT / city / "aoi_prev.gpkg"
    if not prev.exists():
        shutil.copy(ROOT / city / "aoi.gpkg", prev)
    if city not in KEEP_METRO:
        metro[["name", "country_iso", "population", "area_km2", "source", "geometry"]].to_file(ROOT / city / "aoi.gpkg", driver="GPKG")
    core[["name", "country_iso", "population", "area_km2", "source", "geometry"]].to_file(ROOT / city / "core.gpkg", driver="GPKG")
