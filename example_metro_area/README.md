# Example: any metro area

Transit level of service for one area anywhere, from public data.

- **Notebook:** `example.ipynb` runs in Google Colab ([open it in Colab](https://colab.research.google.com/github/CityScope/TransitLOSStudies/blob/main/example_metro_area/example.ipynb)) or Jupyter. Set the place (and the options) in the *Settings* cell, a form in Colab, then Run all: it installs what is missing (a few minutes on the first run, no restart), gets the area, feeds and census, runs the study, shows the interactive map and offers the download. Optional steps, each in its own cell: draw the area on a map or give your own file; untick feeds; set the service date (with a plot of the service on every date); untick groups of census columns; include WorldPop columns. Run all takes the defaults.
- **Script:** `example.py` holds the same steps. Edit the settings at the top and run `python example.py`.

Settings (top of `example.py`):

| Setting | Meaning |
|---|---|
| `PLACE` | place name, e.g. `"Cambridge, Massachusetts, USA"` (the default) |
| `AREA` | `"metro_area"` (UN functional urban area, also `"functional_area"`), `"urban_core"` (UN urban centre) or `"municipality"` |
| `AOI_FILE` | your own boundary file, used instead of `PLACE` |
| `DAY` | `"weekday"` (a working day), `"saturday"` or `"sunday"` |
| `START_TIME`, `END_TIME` | the hours analysed (default `"06:00"` to `"22:00"`) |
| `FEEDS` | `"download"` (find the feeds online), a folder with your GTFS feeds, or a list of zip files / folders / links |
| `FEED_SOURCES` | where `"download"` looks; `None` (default) = the most local sources, then the global ones for the feeds they miss |
| `NAME` | name of the results folder and zip |
| `OUTPUT_DIR` | where the results go (`areas/<name>/`) |
| `CACHE_DIR` | shared downloads (default `TransitLOSStudies/cache`) |
| `BUILD_MAP` | build the interactive map `map.html` (finest hexagons, circles and census, stops and routes; no streets) |

Scoring: the standard transitLOS scoring functions (taken from the transit literature); nothing is fitted to data.

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

Choosing the feeds and the service date (notebook): first the feeds the sources list for the area are shown from their metadata, nothing downloaded
(`study.find_feeds` / `ui.select_candidates`): agency, sources, base city, modes, update date, and whether the coverage includes the area and is not
regional or intercity in size (wider than `COVERAGE_MAX_KM`, 250 km) or the agency is based in the area; those that probably serve the area start ticked. Each feed shows the dates its service is valid when the source gives them (NAP, Transitland), and a search box filters long lists.
Only the ticked feeds are downloaded and listed again with their agencies, source, stops and trips in the area, all ticked (`study.list_feeds(selected=...)`
/ `ui.select_feeds`). The street data is downloaded only when the study runs. The next step plots the service of the chosen feeds on every date (stops × trips in the
area, `study.service_dates` / `ui.pick_date`) with a suggested date (the middle of the typical days of the chosen day type within 60 days of today) and a
date box to use another one. A date you set is used as it is; without one the date is chosen automatically, as in the US study.

Population: with a census whose polygons are small (median area of the finest level at most 1 km²) the census itself provides the population raster
and WorldPop is not downloaded. Coarser or missing censuses use WorldPop as before.

Census sources (notebook): the census comes from several sources, each its own download. Before the run the notebook lists them (`study.census_sources` /
`ui.select_sources`): for the USA the decennial census blocks and the buildings are required, and ACS 5-year, jobs by workplace, religion, housing in
earlier periods, elections and the two WorldPop groups (the population grid, and the age, sex and urban layers: tens of GB) can be ticked or not. The
unticked ones are never downloaded (`CityConfig.census_skip`, and WorldPop only when ticked or when the country has no census). After the run, the columns of the
downloaded sources are grouped by source and can be ticked one by one for the download. The US census needs a free API key (`US_CENSUS_API_KEY`).
