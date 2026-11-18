"""Per-county rescale factors (res 11 group factors of the normal-map rules) recovered from the rebuild logs (`logs/rules/us_*.log`), for
counties built before `county_pipeline` wrote `rescale_factors.json` itself: writes that file per county and `data/rescale_factors.csv`."""
import json, re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PAT = re.compile(r"\[rescale\] county (\d{5}) (\S+): ([\d,.-]+) of ([\d,.-]+) \(([-\d.]+)%\) not on an active cell -- factor ([\d.]+) on (\d+) columns")
rows = []
for log in sorted((ROOT / "logs" / "rules").glob("us_*.log")):
    state = log.stem[3:]
    for m in PAT.finditer(log.read_text(errors="ignore")):
        g, col, miss, tot, share, f, n = m.groups()
        tot = float(tot.replace(",", "")); miss = float(miss.replace(",", ""))
        rows.append({"state": state, "county": g, "anchor": col, "group": "non" if "jobs" in col.lower() else "res", "total": tot,
                     "kept_before_rescale": tot - miss, "share_not_active": float(share) / 100, "factor": float(f)})
df = pd.DataFrame(rows).drop_duplicates(["county", "group"], keep="last")
df.to_csv(ROOT / "data" / "rescale_factors.csv", index=False)
for (state, g), d in df.groupby(["state", "county"]):
    cdir = ROOT / "data" / state / "counties" / g
    if cdir.is_dir():
        fac = {r.group: {"anchor": r.anchor, "total": r.total, "kept_before_rescale": r.kept_before_rescale, "factor": r.factor} for r in d.itertuples()}
        (cdir / "rescale_factors.json").write_text(json.dumps({"note": "original = value / factor (res 11 and every coarser level)", "county": g,
                                                               "factors": fac, "from": "rebuild log"}, indent=1))
print(len(df), "factors,", df["county"].nunique(), "counties; factor range", df["factor"].min(), df["factor"].max())
