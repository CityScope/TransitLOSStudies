"""Write `<city>/results/stops_by_id.parquet`: every real stop_id at its own location with its own routes / mode + its station's scores.

Re-scores each city's stops exactly like the pipeline (same Feed groups, dates from `results/analysis_dates.json`) but does NOT touch the
scores or the street level of service; only the stop-level table is written. Usage: python build_stop_tables.py [city ...]
"""
import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def build(city: str) -> int:
    from code.city_config import CITY_CONFIGS
    from studytools.stop_scoring import score_city_stops

    cfg = CITY_CONFIGS[city]
    d = ROOT / city
    aoi = gpd.read_file(d / "aoi.gpkg")
    dates = {}
    f = d / "results" / "analysis_dates.json"
    if f.is_file():
        dates = {k: date.fromisoformat(v) for k, v in json.loads(f.read_text()).items()}
    date_range = None
    if dates:
        date_range = {k: (v, v) for k, v in dates.items()}
    sink: list = []
    score_city_stops(sorted(str(p) for p in (d / "gtfs").iterdir() if p.is_dir()), aoi, per_feed=cfg.stops_per_feed, date_range=date_range,
                     stop_table_sink=sink)
    table = pd.concat(sink, ignore_index=True) if sink else pd.DataFrame()
    gdf = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table["stop_lon"], table["stop_lat"]), crs=4326) if len(table) else gpd.GeoDataFrame()
    gdf.to_parquet(d / "results" / "stops_by_id.parquet")
    return len(gdf)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        print(c, build(c), flush=True)
