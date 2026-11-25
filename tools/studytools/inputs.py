"""Downloads every study needs: the WorldPop population raster of a country and the OpenStreetMap extract of an area."""

from __future__ import annotations

from pathlib import Path

import requests

#: WorldPop 2025 constrained population, 100 m (R2025A), one GeoTIFF per country (ISO3 code).
WORLDPOP_URL = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/{ISO}/v1/100m/constrained/{iso}_pop_2025_CN_100m_R2025A_v1.tif"


def worldpop_filename(iso3: str) -> str:
    """File name of a country's WorldPop raster (`usa_pop_2025_CN_100m_R2025A_v1.tif`)."""
    return f"{iso3.lower()}_pop_2025_CN_100m_R2025A_v1.tif"


def fetch_file(url: str, dest: Path) -> None:
    """Resumable download to `dest` (via `<dest>.part`, renamed only when complete)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 6):
        try:
            done = part.stat().st_size if part.exists() else 0
            with requests.get(url, stream=True, timeout=120, headers={"Range": f"bytes={done}-"} if done else {}) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                with open(part, "ab" if (done and r.status_code == 206) else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except requests.RequestException as exc:
            print(f"[inputs] attempt {attempt} of {url} failed ({exc})", flush=True)
            if attempt == 5:
                raise
    part.rename(dest)


def worldpop_raster(iso3: str, folder: Path) -> Path:
    """The country's WorldPop raster in `folder`, downloaded when missing."""
    tif = Path(folder) / worldpop_filename(iso3)
    if not tif.is_file():
        print(f"[inputs] downloading WorldPop raster {tif.name}", flush=True)
        fetch_file(WORLDPOP_URL.format(ISO=iso3.upper(), iso=iso3.lower()), tif)
    return tif


#: a census level is fine enough to stand in for WorldPop when the median area of its polygons is at most this (km2)
CENSUS_FINE_MEDIAN_KM2 = 1.0


def census_population_raster(config, aoi, census_dir: Path, tif: Path, max_median_km2: float = CENSUS_FINE_MEDIAN_KM2) -> bool:
    """Write `tif`, a population raster made from the finest census level over `aoi`, to use in place of the WorldPop raster.

    Each census polygon's people are spread evenly over its area (density x pixel area, from a 4x4 supersampling of the polygons), on
    WorldPop's grid (3 arc-seconds, EPSG:4326). Only done when the census is fine enough: it has a population for the area and the
    median polygon is at most `max_median_km2`. Returns False (nothing written) when it is not, so WorldPop should be used.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from .pipeline import _census_population_polygons

    polys, col = _census_population_polygons(config, aoi, census_dir)
    if polys is None or not len(polys):
        return False
    p = polys[[col, "geometry"]].to_crs(4326)
    p = p[p.intersects(aoi.to_crs(4326).union_all())].reset_index(drop=True)
    area_km2 = p.to_crs("EPSG:6933").area.to_numpy() / 1e6
    median = float(np.median(area_km2)) if len(p) else float("inf")
    if not len(p) or median > max_median_km2:
        print(f"[inputs] census polygons too coarse for the area (median {median:.2f} km2 > {max_median_km2} km2): using WorldPop", flush=True)
        return False
    people = np.nan_to_num(p[col].to_numpy(float))
    density = np.where(area_km2 > 0, people / np.where(area_km2 > 0, area_km2, 1.0), 0.0)   # people per km2

    px, sub = 1 / 1200, 4                                                  # WorldPop's pixel (3 arc-seconds); polygon sampling per pixel side
    west, south, east, north = aoi.to_crs(4326).total_bounds
    west, south = np.floor(west / px) * px - px, np.floor(south / px) * px - px
    east, north = np.ceil(east / px) * px + px, np.ceil(north / px) * px + px
    cols, rows = int(round((east - west) / px)), int(round((north - south) / px))
    ids = rasterize(((g, i + 1) for i, g in enumerate(p.geometry)), out_shape=(rows * sub, cols * sub),
                    transform=from_origin(west, north, px / sub, px / sub), fill=0, dtype="int32")
    dens = np.concatenate([[0.0], density])[ids].astype("float32")
    mean = dens.reshape(rows, sub, cols, sub).mean(axis=(1, 3))            # mean density over the pixel (0 where no polygon)
    lat = north - (np.arange(rows) + 0.5) * px
    pixel_km2 = (px * 111.32) ** 2 * np.cos(np.radians(lat))               # pixel area by row
    grid = (mean * pixel_km2[:, None]).astype("float32")
    Path(tif).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif, "w", driver="GTiff", height=rows, width=cols, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(west, north, px, px), compress="deflate") as dst:
        dst.write(grid, 1)
    print(f"[inputs] population raster from the census ({len(p):,} polygons, median {median:.2f} km2, {people.sum():,.0f} people, "
          f"{grid.sum():,.0f} on the grid): WorldPop not needed", flush=True)
    return True


def osm_extract(aoi, folder: Path) -> Path:
    """The smallest Geofabrik `.osm.pbf` extract that covers `aoi` (GeoDataFrame), in `folder`, downloaded when missing."""
    from UrbanAccessAnalyzer.osm_io import download_geofabrik

    return Path(download_geofabrik(aoi, output_folder=str(folder)))
