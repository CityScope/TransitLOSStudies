"""The map's "more housing" / "more transit" development flag, nationwide.

Same definition as the city studies (`city_science_network/code/stats.py::equity_flag_thresholds`): OLS fit of
`level_of_service ~ log(population density)` over the H3 res-7 cells with LOS > 0 and density > 0, then per cell:
  * "more_housing": residual > +1 residual std (better served than its density predicts);
  * "more_transit": residual < -1 residual std AND density > 1000 people/km2.
Here the fit is over every cell of the United States at once (one regression, so the flag is comparable between states). The
fit is saved (`THRESHOLDS_FILE`) so the per-state tile writer (`external_tiles`) flags each state's cells with the same line.
"""
import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd

DENSE_MIN_PER_KM2 = 1000.0
FLAG_RESOLUTION = 7
THRESHOLDS_FILE = Path(__file__).resolve().parent.parent / "data" / "_usa" / "equity_thresholds.json"


def _density(res7: pd.DataFrame) -> np.ndarray:
    area = np.array([h3.cell_area(c, unit="km^2") for c in res7["h3"]])
    return res7["population"].to_numpy(float) / np.where(area > 0, area, np.nan)


def fit_thresholds(res7: pd.DataFrame, density: np.ndarray):
    """`{"slope", "intercept", "std"}` of the national fit, or None when it degenerates."""
    los = res7["level_of_service"].to_numpy(float)
    ok = np.isfinite(density) & np.isfinite(los) & (density > 0) & (los > 0)
    if ok.sum() < 4:
        return None
    x, y = np.log(density[ok]), los[ok]
    slope, intercept = np.polyfit(x, y, 1)
    std = float(np.std(y - (slope * x + intercept)))
    return {"slope": float(slope), "intercept": float(intercept), "std": std} if std > 0 else None


def flags_from_thresholds(res7: pd.DataFrame, thr) -> np.ndarray:
    """Object array of "more_housing" / "more_transit" / None for each row of `res7` (columns h3, population, level_of_service)."""
    flags = np.full(len(res7), None, dtype=object)
    if thr is None:
        return flags
    density = _density(res7)
    los = res7["level_of_service"].to_numpy(float)
    ok = np.isfinite(density) & np.isfinite(los) & (density > 0)
    resid = np.full(len(res7), np.nan)
    resid[ok] = los[ok] - (thr["slope"] * np.log(density[ok]) + thr["intercept"])
    flags[ok & (resid > thr["std"])] = "more_housing"
    flags[ok & (resid < -thr["std"]) & (density > DENSE_MIN_PER_KM2)] = "more_transit"
    return flags


def add_equity_flag(res7: pd.DataFrame) -> pd.DataFrame:
    """Return the national res-7 table with an `equity_flag` column; saves the fit for the per-state tile writer."""
    thr = fit_thresholds(res7, _density(res7))
    THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_FILE.write_text(json.dumps(thr))
    out = res7.copy()
    out["equity_flag"] = flags_from_thresholds(res7, thr)
    return out


def load_thresholds():
    return json.loads(THRESHOLDS_FILE.read_text())
