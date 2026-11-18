"""Per-province rescale factors (res 11 group factors of the normal-map rules) recovered from the rebuild logs (`logs/rules/es_*.log`), for
provinces built before `province_pipeline` wrote `rescale_factors.json` itself: `data/rescale_factors.csv` (latest run per province)."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PAT = re.compile(r"^(\S+ \S+) .*\[rescale\] province (\d{2}) (\S+): ([\d,.-]+) of ([\d,.-]+) \(([-\d.]+)%\) not on an active cell -- factor ([\d.]+) on (\d+) columns", re.M)
rows = []
for log in sorted((ROOT / "logs" / "rules").glob("es_*.log")):
    for m in PAT.finditer(log.read_text(errors="ignore")):
        t, p, col, miss, tot, share, f, n = m.groups()
        tot = float(tot.replace(",", "")); miss = float(miss.replace(",", ""))
        rows.append({"time": t, "community": log.stem[3:], "province": p, "anchor": col, "total": tot, "kept_before_rescale": tot - miss,
                     "share_not_active": float(share) / 100, "factor": float(f), "columns": int(n)})
df = pd.DataFrame(rows)
df = df[df["anchor"] == "population"]   # building stock is no longer rescaled (2026-11-13)
df = df.sort_values("time").drop_duplicates(["province", "anchor"], keep="last")
df.to_csv(ROOT / "data" / "rescale_factors.csv", index=False)
print(len(df), "factors,", df["province"].nunique(), "provinces; factor range", df["factor"].min(), df["factor"].max())
