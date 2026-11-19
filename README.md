# TransitLOSStudies

Transit level-of-service studies built on the `transitLOS`, `pyGTFSHandler`, `pyCensus`, `geohierarchy` and `UrbanAccessAnalyzer`
packages:

- `city_science_network/`: 11 city maps (Andorra, Beer Sheva, Boston, Concepción, Gipuzkoa, Guadalajara, Hamburg, San Francisco,
  Shanghai, Taipei, Toronto);
- `spain_study/`, `usa_study/`: the nationwide Spain and US maps;
- `usa_flex/`: the US flexible-scoring map and the score fits against observed transit use;
- `example_metro_area/`: the same study for any area you choose, as a notebook (`example.ipynb`, runs in Google Colab) and a script.

The maps and the docs: https://cityscope.github.io/TransitLOSStudies/. Any other area: open
[`example_metro_area/example.ipynb` in Google Colab](https://colab.research.google.com/github/CityScope/TransitLOSStudies/blob/main/example_metro_area/example.ipynb).

The repository holds the code and the inputs that can no longer be downloaded (archived feeds, hand-built feeds with their
`ASSUMED.md`, analysis areas, feed lists, the fits' warm start). Everything else is downloaded into the shared cache (`cache/`,
or `STUDY_CACHE_DIR`) and regenerated.

## Running the studies

Each study folder has its own `run.sh`, which builds everything in it from the code, the inputs kept here and public downloads:

| Study | Command | Writes |
|---|---|---|
| City maps | `city_science_network/run.sh [city ...]` | `city_science_network/<city>/map.html` |
| Spain map | `spain_study/run.sh` | `spain_study/data/_spain/` |
| US map | `usa_study/run.sh` | `usa_study/data/_usa/` |
| US flexible-scoring map + fit data | `usa_flex/run.sh` (after the US map) | `usa_flex/web_buf/`, `usa_flex/fit/data_buf/` |
| Score study | `usa_flex/fit/score_study/ctrl_study7.sh`, then `ctrl_study7b.sh` | `usa_flex/fit/data_buf/torch_runs/` |
| Any area | `example_metro_area/example.ipynb` or `python example_metro_area/example.py` | `example_metro_area/areas/<name>/` |

Each `run.sh` checks its maps at the end with the shared `checks/` (`check_map_data.py`: the data files; `test_map.py`: the page
opens and draws, with a screenshot; needs Playwright), writes its logs to `logs/` in its folder (`LOGS=` to change it) and resumes
where it stopped. The US and Spain builds run one memory-capped process at a time (70 GB). The score fits need a Python with
PyTorch (`TORCH_PYTHON=/path/to/python`) and a GPU with ~22 GB free, so map builds running at the same time use
`TRANSITLOS_DEVICE=cpu`. API keys: see the pyGTFSHandler and pyCensus agent guides (one file, `~/.config/ptlos/api_keys.json`);
the two Toronto census files that only a browser can download are listed in the pyCensus guide.

Shared by the studies: `tools/studytools/` (the per-area pipeline, stop scoring, shared cache and downloads; tests in
`tools/tests/`), `checks/`, `memsched.py` (memory-capped parallel runs) and `study_cache_shared.py`. Anything used by one study
only lives in that study's folder, and no study imports another study's code.

Tests: `python -m pytest` in `tools/`, `city_science_network/` and `usa_study/`.

## Large files

A pre-commit hook (`tools/check_large_files.py`, installed with `python3 tools/check_large_files.py --install`) refuses a commit with a file over 10 MB once: run the same commit again to keep it. Files over 50 MB are never committed.

## Local files

This repository is public. API keys, notes about a particular machine or its processes (hosting, uploads, local paths) and
upload scripts are never committed: keep notes as `*.local.md` or under `local/`, which `.gitignore` excludes.
