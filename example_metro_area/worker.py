"""Runs one function of `example.py` in the study environment: `python worker.py '<json>'` (called by `bootstrap.call`).

Areas travel as small dicts (`name`, `km2`, `geojson`, `outline`, `bounds`) so the notebook's own kernel needs no geopandas.
"""
import json
import sys
import warnings
from pathlib import Path

# pandas warns when an optional speed-up library (bottleneck, numexpr) is older than it likes, as Colab's are; the results do not change
warnings.filterwarnings("ignore", message=r"Pandas requires version .* or newer of '.*' \(version .* currently installed\)", category=UserWarning)

# the run prints thousands of harmless library warnings (fragmented frames, centroids in degrees, divisions by zero in shares): left out of the notebook's output
warnings.filterwarnings("ignore", message=r"DataFrame is highly fragmented")
warnings.filterwarnings("ignore", message=r"Geometry is in a geographic CRS")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"(divide by zero|invalid value|overflow) encountered")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _in(v):
    import geopandas as gpd
    from shapely.geometry import shape

    if isinstance(v, dict) and "geojson" in v:
        return gpd.GeoDataFrame({"name": [v["name"]]}, geometry=[shape(v["geojson"])], crs=4326)
    if isinstance(v, (list, tuple)):
        return [_in(x) for x in v]
    return v


def _out(v):
    import geopandas as gpd

    if isinstance(v, gpd.GeoDataFrame):
        geom = v.to_crs(4326).union_all()
        minx, miny, maxx, maxy = geom.bounds
        outline = geom.simplify(max(maxx - minx, maxy - miny) / 500)
        import example

        return {"name": str(v["name"].iloc[0]), "km2": example.area_km2(v), "geojson": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
                "outline": json.loads(gpd.GeoSeries([outline], crs=4326).to_json()), "bounds": [minx, miny, maxx, maxy]}
    return str(v) if isinstance(v, Path) else v


if __name__ == "__main__":
    request = json.loads(sys.argv[1])
    import example

    result = getattr(example, request["fn"])(*_in(request["args"]), **{k: _in(v) for k, v in request["kwargs"].items()})
    Path(request["out"]).write_text(json.dumps(_out(result)))
