"""Write the four analysis areas of every city-science city and the processing area (their union).

    metro_official.gpkg  official metropolitan area: every municipality that belongs to it officially (list and source in OFFICIAL below)
    municipality.gpkg    the municipality that gives the city its name
    un_metro.gpkg        UN Degree of Urbanisation functional urban area (pycensus.worldwide.un_metro_areas), no holes
    core.gpkg            UN urban centre (Andorra: see scripts_un_areas.py), no holes
    aoi.gpkg             union of the four (the area that is processed); the previous file is kept as aoi_prev_areas.gpkg

Run from city_science_network/: `python scripts_four_areas.py [--dry] [city ...]`.
Sources: GADM 4.1 municipal boundaries (cache/gadm), US Census cartographic boundaries 2023 (CBSA, place), the cities' previous official-metro files
(aoi_prev.gpkg: Boston = OMB CBSA 14460, Concepcion = Decreto 326/2023 (11 comunas), Hamburg = Metropolregion Hamburg members, Taipei, Toronto) and
Nominatim administrative polygons (Israel).
"""
import io, re, shutil, sys, zipfile
from pathlib import Path
import geopandas as gpd, pandas as pd
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "tools"))
from pycensus.worldwide import un_metro_areas as un
from studytools.city_core import _geocode_admin_polygon

CACHE = ROOT.parent / "cache"
GADM = CACHE / "gadm"
import tempfile
TMP = Path(tempfile.gettempdir())
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DRY = "--dry" in sys.argv
EQ = "EPSG:6933"
POINTS = {"toronto": (-79.38, 43.65), "boston": (-71.06, 42.36), "san_francisco": (-122.42, 37.77), "hamburg": (9.99, 53.55), "taipei": (121.56, 25.04),
          "gipuzkoa": (-1.98, 43.32), "beerseba": (34.79, 31.25), "concepcion": (-73.05, -36.82), "guadalajara": (-103.35, 20.67), "shanghai": (121.47, 31.23),
          "andorra": (1.52, 42.51)}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n").replace("ü", "u"))


def gadm(code, col, names, **flt):
    g = gpd.read_file(GADM / f"gadm41_{code}.json").to_crs(4326)
    for k, v in flt.items():
        g = g[g[k].map(norm) == norm(v)]
    want = {norm(n) for n in names}
    sel = g[g[col].map(norm).isin(want)]
    missing = want - set(sel[col].map(norm))
    assert not missing, f"{code}: not found {missing}"
    return unary_union(sel.geometry.values)


def previous(city, name=None):
    g = gpd.read_file(ROOT / city / "aoi_prev.gpkg").to_crs(4326)
    if name is not None:
        col = next(c for c in ("municipality", "Kreis", "name") if c in g.columns)
        g = g[g[col] == name]
        assert len(g) == 1, (city, name, len(g))
    return unary_union(g.geometry.values)


def us_geom(level, state, geoid, year=2023):
    """Official US area from pycensus's geometry loader (TIGER/Line): CBSA or place."""
    from pycensus.countries.usa.geography import load_boundaries
    g = load_boundaries(level, state, year, erase_water=True)
    col = next(c for c in ("GEOID", "GEOIDFQ") if c in g.columns)
    sel = g[g[col].astype(str).str.endswith(geoid)]
    assert len(sel) >= 1, (level, state, geoid)
    return unary_union(sel.to_crs(4326).geometry.values)


def tiger(url, name, sel):
    z = TMP / name
    if not z.is_file():
        import requests
        z.write_bytes(requests.get(url, timeout=600).content)
    d = TMP / (name + "_x")
    zipfile.ZipFile(z).extractall(d)
    g = gpd.read_file(next(d.glob("*.shp"))).to_crs(4326)
    return unary_union(g[sel(g)].geometry.values)


def osm(query):
    return _geocode_admin_polygon(query).to_crs(4326).geometry.iloc[0]


def hole_free(geom):
    from shapely.geometry import Polygon, MultiPolygon
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])


# (official metro, municipality, source text of the official metro area)
def official(city):
    if city == "andorra":
        return gadm("AND_1", "COUNTRY", ["Andorra"]), gadm("AND_1", "NAME_1", ["Andorra la Vella"]), "Principality of Andorra (whole country)"
    if city == "beerseba":
        # CBS metropolitan area of Be'er Sheva (core + outer ring); the outer ring localities are the municipalities / local and regional councils below.
        # APPROXIMATION: CBS lists 32 outer-ring localities (many unrecognised Bedouin villages inside the regional councils), not boundaries.
        def _try(q):
            try:
                return osm(q)
            except Exception as exc:   # an authority Nominatim has no polygon for is skipped (listed in the printed output)
                print("   skipped", q, str(exc)[:60]); return None
        parts = [_try(q) for q in ("Beersheba, Israel", "Omer, Israel", "Lehavim, Israel", "Meitar, Israel", "Tel Sheva, Israel", "Rahat, Israel", "Kuseife, Israel",
                                  "Hura, Israel", "Lakiya, Israel", "Segev Shalom, Israel", "Arara BaNegev, Israel", "Shaqib al-Salam, Israel",
                                  "Bnei Shimon Regional Council, Israel", "Neve Midbar Regional Council, Israel")]
        parts = [p for p in parts if p is not None]
        muni = gpd.read_file(ROOT / "beerseba" / "aoi_city.gpkg").to_crs(4326).union_all()
        return unary_union(parts), muni, "CBS Beer Sheva metropolitan area: Beersheba + outer-ring local authorities"
    if city == "boston":
        return us_geom("core_based_statistical_area", "MA", "14460"), us_geom("place", "MA", "2507000"), "OMB Boston-Cambridge-Newton MA-NH metropolitan statistical area (CBSA 14460), pycensus geometry loader"
    if city == "concepcion":
        # a whole-region official area (Region del Biobio, 33 comunas) was tried and dropped: no usable route/frequency data exists outside Gran Concepcion
        return previous(city), previous(city, "Concepción"), "Area Metropolitana del Gran Concepcion, Decreto 326/2023 Ministerio del Interior (11 comunas)"
    if city == "gipuzkoa":
        # the whole province (GADM ESP_2 Guipuzcoa, coastline-clipped); the 89 geocoded municipality polygons of aoi_prev.gpkg unioned into a
        # coarse outline reaching the sea, Bizkaia and Navarra (2026-11-12)
        g4 = gpd.read_file(GADM / "gadm41_ESP_4.json").to_crs(4326)
        prov = unary_union(g4[g4["NAME_2"].map(norm) == norm("Guipúzcoa")].geometry.values)   # all its municipalities (GADM ESP_4)
        return prov, gadm("ESP_4", "NAME_4", ["Donostia-San Sebastián"], NAME_2="Guipúzcoa"), "Gipuzkoa province (Territorio Historico de Gipuzkoa)"
    if city == "guadalajara":
        names = ["El Salto", "Guadalajara", "Ixtlahuacán de los Membrillos", "Juanacatlán", "San Pedro Tlaquepaque", "Tlajomulco de Zúñiga", "Tonalá", "Zapopan", "Zapotlanejo"]
        # a whole-state official area (Jalisco, 125 municipalities) was tried and dropped: OSM has bus routes only for the AMG and Puerto Vallarta, no feeds elsewhere
        return gadm("MEX_2", "NAME_2", names, NAME_1="Jalisco"), gadm("MEX_2", "NAME_2", ["Guadalajara"], NAME_1="Jalisco"), \
            "Area Metropolitana de Guadalajara, Decreto 23021/LVIII/09 and 25400/LX/15 (9 municipalities)"
    if city == "hamburg":
        return previous(city), previous(city, "Hamburg"), "Metropolregion Hamburg members (Hamburg + 20 districts and cities in 4 Laender)"
    if city == "san_francisco":
        msa = us_geom("core_based_statistical_area", "CA", "41860")
        sf = us_geom("place", "CA", "0667000")
        return msa, sf, "OMB San Francisco-Oakland-Berkeley CA metropolitan statistical area (CBSA 41860), pycensus geometry loader"
    if city == "shanghai":
        names = ["Shanghai", "Suzhou", "Wuxi", "Changzhou", "Nantong", "Ningbo", "Jiaxing", "Huzhou", "Zhoushan"]
        # "Suzhou" also names Suzhou (宿州) in Anhui, 800 km away: the Jiangsu one is selected by province
        official = unary_union([gadm("CHN_2", "NAME_2", [n for n in names if n != "Suzhou"]), gadm("CHN_2", "NAME_2", ["Suzhou"], NAME_1="Jiangsu")])
        return official, gadm("CHN_2", "NAME_2", ["Shanghai"]), "Shanghai Metropolitan Area (1+8) spatial coordination plan 2022: Shanghai, Suzhou, Wuxi, Changzhou, Nantong, Ningbo, Jiaxing, Huzhou, Zhoushan"
    if city == "taipei":
        return previous(city), gadm("TWN_2", "NAME_2", ["Taipei"]), "Taipei metropolitan area: Taipei, New Taipei, Keelung and Taoyuan"
    if city == "toronto":
        return previous(city), previous(city, "Toronto"), "Toronto census metropolitan area / Greater Toronto Area census subdivisions"
    raise KeyError(city)


def andorra_un_style():
    """UN Degree of Urbanisation classes computed from the WorldPop 100 m population grid (the GHSL UN datasets have no entry for Andorra).

    1 km grid (EPSG:3035, sum of the 100 m cells); urban centre = contiguous (4-neighbour) cells >= 1,500 inhabitants/km2 with >= 50,000 inhabitants in
    total; urban cluster = the same with >= 300/km2 and >= 5,000 inhabitants. Andorra's conurbation is below the urban-centre size, so the UN core is
    the urban cluster that holds Andorra la Vella. The UN functional area (urban centre + commuting zone) cannot be built without an urban centre: the
    whole country is used instead (every commune is within commuting distance of the capital).
    """
    import numpy as np, rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio import features
    from scipy import ndimage
    from shapely.geometry import shape, Point
    import glob
    tif = glob.glob(str(CACHE / "worldpop" / "**" / "and_pop_2025_CN_100m_R2025A_v1.tif"), recursive=True)[0]
    with rasterio.open(tif) as src:
        res = 1000.0
        bounds = gpd.GeoSeries([gpd.GeoSeries.from_xy([1.40], [42.42], crs=4326).iloc[0], gpd.GeoSeries.from_xy([1.80], [42.66], crs=4326).iloc[0]], crs=4326).to_crs(3035).total_bounds
        x0, y0, x1, y1 = np.floor(bounds[0] / res) * res, np.floor(bounds[1] / res) * res, np.ceil(bounds[2] / res) * res, np.ceil(bounds[3] / res) * res
        w, h = int((x1 - x0) / res), int((y1 - y0) / res)
        tr = rasterio.transform.from_origin(x0, y1, res, res)
        pop = np.zeros((h, w), dtype="float64")
        reproject(rasterio.band(src, 1), pop, src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs="EPSG:3035", resampling=Resampling.sum, dst_nodata=0)
    pop = np.nan_to_num(np.where(pop < 0, 0, pop))
    cap = gpd.GeoSeries.from_xy([1.5218], [42.5063], crs=4326).to_crs(3035).iloc[0]
    out = {}
    for name, dens, tot in (("centre", 1500, 50000), ("cluster", 300, 5000)):
        lab, n = ndimage.label(pop >= dens)
        r, c = rasterio.transform.rowcol(tr, cap.x, cap.y)
        k = lab[r, c]
        if k == 0:   # capital cell below the density threshold: use the densest neighbouring cell's cluster
            win = lab[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]; k = win.max()
        total = pop[lab == k].sum() if k else 0
        print(f"   andorra {name}: cluster pop {total:,.0f} (needs {tot:,})")
        if k and total >= tot:
            polys = [shape(g) for g, v in features.shapes((lab == k).astype("uint8"), mask=(lab == k), transform=tr)]
            out[name] = gpd.GeoSeries([unary_union(polys)], crs=3035).to_crs(4326).iloc[0]
    geom = out.get("centre") or out["cluster"]
    return geom, ("urban centre" if "centre" in out else "urban cluster")


def un_areas(city):
    pt = POINTS[city]
    if city == "andorra":
        geom, kind = andorra_un_style()
        core = gpd.GeoDataFrame({"name": [f"Andorra la Vella {kind}"]}, geometry=[geom], crs=4326)
        metro = gpd.GeoDataFrame({"name": ["Andorra (no UN functional area: whole country)"]}, geometry=[gadm("AND_1", "COUNTRY", ["Andorra"])], crs=4326)
        return metro, core
    metro = un.find(point=pt, kind="functional_urban_area", cache_dir=str(CACHE / "pycensus")).iloc[0:1]
    core = un.find(point=pt, kind="urban_centre", cache_dir=str(CACHE / "pycensus")).iloc[0:1]
    return metro, core


def exclude_palestine(geom):
    import geopandas as gp
    ne = gp.read_file(CACHE / "natural_earth" / "ne_10m_admin_0_countries.shp")
    return geom.difference(ne[ne["ADM0_A3"] == "PSX"].to_crs(4326).union_all())


def save(geom, path, name, source):
    Path(path).unlink(missing_ok=True)   # a file with several layers would keep its old default layer
    gpd.GeoDataFrame({"name": [name], "source": [source]}, geometry=[hole_free(geom)], crs=4326).to_file(path, driver="GPKG")


#: display names of the areas: the official area and the other three (municipality, UN metro, UN core) carry these names when they differ from the defaults
OFFICIAL_NAME = {"gipuzkoa": "Gipuzkoa"}
CITY_NAME = {"concepcion": "Concepción", "gipuzkoa": "Donostia", "guadalajara": "Guadalajara"}

for city in (ARGS or list(POINTS)):
    d = ROOT / city
    metro_off, muni, src = official(city)
    um, uc = un_areas(city)
    um_g, uc_g = um.geometry.iloc[0], uc.geometry.iloc[0]
    if city == "beerseba":
        metro_off, muni, um_g, uc_g = (exclude_palestine(x) for x in (metro_off, muni, um_g, uc_g))
    if city == "andorra":   # core of the study: Andorra la Vella; the UN centre stays as the UN area
        pass
    union = unary_union([metro_off, muni, um_g, uc_g])
    cur = gpd.read_file(d / "aoi.gpkg").to_crs(4326).union_all()
    a = lambda g: gpd.GeoSeries([g], crs=4326).to_crs(EQ).iloc[0].area / 1e6
    print(f"{city:14s} official metro {a(metro_off):8.0f} | municipality {a(muni):6.0f} | UN metro {a(um_g):7.0f} | UN core {a(uc_g):6.0f} | union {a(union):8.0f} | current aoi {a(cur):8.0f}"
          f" | union outside current {a(union.difference(cur)) / a(union):5.1%}", flush=True)
    if DRY:
        gpd.GeoDataFrame({"name": [city]}, geometry=[union], crs=4326).to_file(TMP / f"union_{city}.gpkg", driver="GPKG")
        continue
    if not (d / "aoi_prev_areas.gpkg").exists():
        shutil.copy(d / "aoi.gpkg", d / "aoi_prev_areas.gpkg")
    save(metro_off, d / "metro_official.gpkg", OFFICIAL_NAME.get(city, "Metro area"), src)
    save(muni, d / "municipality.gpkg", CITY_NAME.get(city, "Municipality"), "official municipality")
    if city == "andorra":   # no GHS-FUA / GHS-UCDB entry: computed with the UN Degree of Urbanisation method on the WorldPop 1 km grid (see andorra_un_style)
        save(um_g, d / "un_metro.gpkg", um["name"].iloc[0], "UN-style (no GHS-FUA entry for Andorra): whole country as the functional area")
        save(uc_g, d / "core.gpkg", uc["name"].iloc[0], "UN-style (no GHS-UCDB entry for Andorra): Degree of Urbanisation urban cluster (>=300 inh/km2, >=5,000 inh) from the WorldPop 1 km grid; the urban centre (>=1,500 inh/km2) holds 32,466 people, below the 50,000 minimum")
    else:
        save(um_g, d / "un_metro.gpkg", CITY_NAME.get(city, um["name"].iloc[0]), "UN Degree of Urbanisation: functional urban area (GHS-FUA R2019A)")
        save(uc_g, d / "core.gpkg", CITY_NAME.get(city, uc["name"].iloc[0]), "UN Degree of Urbanisation: urban centre (GHS-UCDB R2024A)")
    save(union, d / "aoi.gpkg", f"{city} processing area", "union of metro_official, municipality, un_metro, core")
