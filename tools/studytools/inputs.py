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


def osm_extract(aoi, folder: Path) -> Path:
    """The smallest Geofabrik `.osm.pbf` extract that covers `aoi` (GeoDataFrame), in `folder`, downloaded when missing."""
    from UrbanAccessAnalyzer.osm_io import download_geofabrik

    return Path(download_geofabrik(aoi, output_folder=str(folder)))
