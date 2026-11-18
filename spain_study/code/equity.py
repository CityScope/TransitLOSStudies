"""The map's "more housing" / "more transit" development flag, ONE REGRESSION PER PROVINCE, computed at H3 res 8.

Same definition as the city studies (`city_science_network/code/stats.py::equity_flag_thresholds`, which fits one line per city), applied
inside each province on its own res-8 cells (`provinces/<code>/h3_res8.geoparquet`):
  * OLS fit of `level_of_service ~ log(population density)` over the province's res-8 cells with LOS > 0 and density > 0;
  * "more_housing": residual > +1 residual std of that province's fit (better served than its density predicts);
  * "more_transit": residual < -1 residual std AND density > `DENSE_MIN_PER_KM2` (1000 people/km2; density is per km2, so it does not depend on
    the cell size);
  * a province with fewer than 4 usable cells (or a degenerate fit) gets no flags.
There is no national line: each community folder stores its provinces' fits in `<community>/equity_thresholds.json`
(`{province GEOID: {"slope", "intercept", "std"} | null}`), used by the map inputs (`add_equity_flag`) and by the per-community development tile writer
(`external_tiles`). Re-apply to existing outputs: `python -c "from code.equity import write_all_thresholds; write_all_thresholds()"` from `spain_study/`,
then rebuild the map (page_only rebuild re-tiles the development overlay).
"""
import json
from pathlib import Path
from typing import Iterable, Optional

import h3
import numpy as np
import pandas as pd

DENSE_MIN_PER_KM2 = 1000.0
FLAG_RESOLUTION = 8
GROUP_DIRNAME = "provinces"
THRESHOLDS_NAME = "equity_thresholds.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _density(table: pd.DataFrame) -> np.ndarray:
    area = np.array([h3.cell_area(c, unit="km^2") for c in table["h3"]])
    return table["population"].to_numpy(float) / np.where(area > 0, area, np.nan)


def fit_thresholds(table: pd.DataFrame, density: np.ndarray):
    """`{"slope", "intercept", "std"}` of one group's fit, or None when it degenerates (< 4 finite cells, std 0)."""
    los = table["level_of_service"].to_numpy(float)
    ok = np.isfinite(density) & np.isfinite(los) & (density > 0) & (los > 0)
    if ok.sum() < 4:
        return None
    x, y = np.log(density[ok]), los[ok]
    slope, intercept = np.polyfit(x, y, 1)
    std = float(np.std(y - (slope * x + intercept)))
    return {"slope": float(slope), "intercept": float(intercept), "std": std} if std > 0 else None


def flags_from_thresholds(table: pd.DataFrame, thr) -> np.ndarray:
    """Object array of "more_housing" / "more_transit" / None for each row of `table` (columns h3, population, level_of_service)."""
    flags = np.full(len(table), None, dtype=object)
    if thr is None or not len(table):
        return flags
    density = _density(table)
    los = table["level_of_service"].to_numpy(float)
    ok = np.isfinite(density) & np.isfinite(los) & (density > 0)
    resid = np.full(len(table), np.nan)
    resid[ok] = los[ok] - (thr["slope"] * np.log(density[ok]) + thr["intercept"])
    flags[ok & (resid > thr["std"])] = "more_housing"
    flags[ok & (resid < -thr["std"]) & (density > DENSE_MIN_PER_KM2)] = "more_transit"
    return flags


def group_tables(state_dir: Path) -> Iterable[tuple]:
    """`(group id, res-8 table[h3, population, level_of_service])` for every province of the community folder."""
    for d in sorted((Path(state_dir) / GROUP_DIRNAME).iterdir()):
        f = d / f"h3_res{FLAG_RESOLUTION}.geoparquet"
        if f.is_file():
            yield d.name, pd.read_parquet(f, columns=["h3", "population", "level_of_service"])


def state_thresholds(state_dir: Path, write: bool = True) -> dict:
    """`{group id: fit | None}` for one community folder (written to `<community>/equity_thresholds.json`)."""
    thr = {gid: fit_thresholds(t, _density(t)) for gid, t in group_tables(state_dir)}
    if write:
        (Path(state_dir) / THRESHOLDS_NAME).write_text(json.dumps(thr))
    return thr


def load_thresholds(state_dir: Path) -> dict:
    """The state's per-province fits (computed and saved when the file does not exist yet)."""
    f = Path(state_dir) / THRESHOLDS_NAME
    return json.loads(f.read_text()) if f.is_file() else state_thresholds(state_dir)


def state_flagged_cells(state_dir: Path, only_flagged: bool = True) -> pd.DataFrame:
    """Res-8 cells of every province of the state with `equity_flag` from THEIR province's own line (`h3`, population, level_of_service, province)."""
    thr = load_thresholds(state_dir)
    parts = []
    for gid, t in group_tables(state_dir):
        t = t.copy()
        t["equity_flag"] = flags_from_thresholds(t, thr.get(gid))
        t["province"] = gid
        parts.append(t[t["equity_flag"].notna()] if only_flagged else t)
    if not parts:
        return pd.DataFrame(columns=["h3", "population", "level_of_service", "equity_flag", "province"])
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("population", ascending=False).drop_duplicates("h3")   # a border cell listed by two counties: its larger share wins


def add_equity_flag(res8: pd.DataFrame, state_dirs: Optional[Iterable[Path]] = None) -> pd.DataFrame:
    """`res8` (the map's res-8 table, column `h3`) with `equity_flag` from each cell's own province fit. Lazy: only reads the counties' res-8 files."""
    flagged = pd.concat([state_flagged_cells(d, only_flagged=False) for d in (state_dirs or [])], ignore_index=True) if state_dirs else None
    out = res8.copy()
    if flagged is None or flagged.empty:
        out["equity_flag"] = None
        return out
    flagged = flagged.sort_values("population", ascending=False).drop_duplicates("h3")
    out["equity_flag"] = out["h3"].map(flagged.set_index("h3")["equity_flag"])
    out["equity_flag"] = out["equity_flag"].astype(object).where(out["equity_flag"].notna(), None)
    return out


def write_all_thresholds(data_dir: Path = DATA_DIR) -> int:
    """Recompute every state's `equity_thresholds.json` from the finished province outputs; returns the number of states."""
    n = 0
    for d in sorted(Path(data_dir).iterdir()):
        if (d / GROUP_DIRNAME).is_dir():
            state_thresholds(d)
            n += 1
    return n
