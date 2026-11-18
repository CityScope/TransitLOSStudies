"""Pack the GTFS feeds that `code/gtfs_fetch.py` cannot download by itself into `<city>/gtfs_archive/` so that they can live in the repository.

    taipei/tdx_bus, taipei/trtc_metro   TDX needs an account
    shanghai/shanghai_metro_area        output of the Shanghai builder (kept so a run does not depend on re-running it)
    andorra/ena_manual_reconstruction   output of the Andorra builder

Each archive is `<feed>.zip`; one that would exceed MAX_PART (50 MB limit of the repository, 45 MB used) is split into `<feed>.zip.001`, `.002`, ... .
`code/gtfs_fetch.restore_archives` joins and extracts them into `<city>/gtfs/<feed>/` when that folder is missing. The TDX fare files (fare_leg_rules,
fare_attributes, fare_media, rider_categories, translations, areas, stop_areas; 4 GB, never read by the study) are left out.
Run from city_science_network/: `python archive_feeds.py`. Nothing is committed by this script.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_PART = 45 * 1024 * 1024
FEEDS = [("taipei", "tdx_bus"), ("taipei", "trtc_metro"), ("shanghai", "shanghai_metro_area"), ("andorra", "ena_manual_reconstruction")]
SKIP = {"fare_leg_rules.txt", "fare_attributes.txt", "fare_media.txt", "rider_categories.txt", "translations.txt", "areas.txt", "stop_areas.txt"}

for city, feed in FEEDS:
    src = ROOT / city / "gtfs" / feed
    if not src.is_dir():
        print("missing", src); continue
    out_dir = ROOT / city / "gtfs_archive"; out_dir.mkdir(exist_ok=True)
    zpath = out_dir / f"{feed}.zip"
    for old in out_dir.glob(f"{feed}.zip*"):
        old.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(src.iterdir()):
            if f.is_file() and f.name not in SKIP:
                z.write(f, f.name)
    size = zpath.stat().st_size
    if size > MAX_PART:
        data_parts = 0
        with open(zpath, "rb") as fh:
            while True:
                chunk = fh.read(MAX_PART)
                if not chunk:
                    break
                data_parts += 1
                (out_dir / f"{feed}.zip.{data_parts:03d}").write_bytes(chunk)
        zpath.unlink()
        print(f"{city}/{feed}: {size / 1e6:.0f} MB in {data_parts} parts")
    else:
        print(f"{city}/{feed}: {size / 1e6:.1f} MB")
