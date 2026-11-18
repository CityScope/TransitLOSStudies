# Example: any metro area

Transit level of service for one area anywhere, from public data.

- **Notebook:** `example.ipynb` runs in Google Colab or Jupyter. It installs what is missing, lets you choose the area (place name, drawn on a map, or uploaded file), runs the study and downloads the results.
- **Script:** `example.py` holds the same steps. Edit the settings at the top and run `python example.py`.

Settings (top of `example.py`):

| Setting | Meaning |
|---|---|
| `PLACE` | place name, e.g. `"Boston, Massachusetts, USA"` (the default) |
| `AREA` | `"municipality"`, `"urban_core"` (UN urban centre) or `"metro_area"` (UN metropolitan area) |
| `AOI_FILE` | your own boundary file, used instead of `PLACE` |
| `DAY` | `"weekday"` (a working day), `"saturday"` or `"sunday"` |
| `START_TIME`, `END_TIME` | the hours analysed (default `"06:00"` to `"22:00"`) |
| `FEEDS` | `"download"` (find the feeds online), a folder with your GTFS feeds, or a list of zip files / folders / links |
| `FEED_SOURCES` | where `"download"` looks; `None` (default) = the most local sources, then the global ones for the feeds they miss |
| `NAME` | name of the results folder and zip |
| `OUTPUT_DIR` | where the results go (`areas/<name>/`) |
| `CACHE_DIR` | shared downloads (default `TransitLOSStudies/cache`) |
| `BUILD_MAP` | build the interactive map `map.html` (finest hexagons, circles and census, stops and routes; no streets) |

What it uses: every input from the most local source that covers the area (the `*_BY_PLACE` tables in `example.py`; add a better
local source above the general ones):

| Input | Most local first |
|---|---|
| Census | the region's census (Basque Country), then the country's (USA, Canada, Mexico, Spain, Germany, Israel, Andorra, Taiwan, Chile), then WorldPop. With a census the results have no WorldPop columns. |
| Transit feeds | the country's official portal (US National Transit Database, Spain's National Access Point), plus the feeds it misses from the Mobility Database catalog and Transitland (copies of an official feed are dropped). Or your own feeds. |
| Municipality boundary | the country's (US Census places), then OpenStreetMap |
| Urban core, metro area | UN Degree of Urbanisation (GHSL) |
| Streets | OpenStreetMap (Geofabrik) |

API keys: when a source needs a key that is missing, the run says where to sign up and asks for it (Enter skips that source).
Keys are saved in `~/.config/ptlos/api_keys.json`.

Results (`areas/<name>/<name>_results.zip`):

- `streets.parquet`: street segments with their transit level of service.
- `h3_res<N>.parquet`: H3 hexagons at each resolution, with population, level of service and the census columns.
- `census_<level>.parquet`: the census at its finest level.
