"""Cross-level consistency of the 2024 president columns in every state's data: statewide sum on each level vs the FEC official total.

Run after the election patch + H3 rebuild:  python check_elections.py   (exit 1 when any level of any state is off by more than 0.05%).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
import pycensus  # noqa: E402

FEC = json.load(open(Path(pycensus.__file__).parent / "countries/usa/elections_precinct/unofficial/verification/fec_2024_state_totals.json"))["states"]

# only levels that tile the whole state (places and school districts cover part of it by definition); H3 may lose coastal cells
EXACT = ("county", "voting_district", "congressional_district", "state_legislative_district")
H3 = ("h3_res8", "h3_res5")
LEVELS = EXACT + H3
COL = "2024_president_total_votes"
STATE_AB = {n: a.lower() for a, n in {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "DistrictofColumbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "NewHampshire", "NJ": "NewJersey", "NM": "NewMexico", "NY": "NewYork", "NC": "NorthCarolina", "ND": "NorthDakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "RhodeIsland", "SC": "SouthCarolina", "SD": "SouthDakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "WestVirginia", "WI": "Wisconsin", "WY": "Wyoming"}.items()}
bad = 0
for d in sorted((HERE / "data").iterdir()):
    if not (d / "state.geoparquet").exists():
        continue
    st_total = float(pd.read_parquet(d / "state.geoparquet", columns=[COL]).iloc[0, 0])
    out = {"state": d.name, "state_layer": st_total}
    for lv in LEVELS:
        p = d / f"{lv}.geoparquet"
        if p.exists() and COL in pq.read_schema(p).names:
            out[lv] = float(pd.read_parquet(p, columns=[COL])[COL].sum())
    ref = FEC[STATE_AB[d.name]]["total"]
    worst = max([abs(out[k] / ref - 1) for k in ("state_layer",) + EXACT if k in out] + [abs(out[k] / ref - 1) / 10 for k in H3 if k in out] + [0])
    spread = worst
    flag = "OK" if worst < 1e-3 else "INCONSISTENT"
    bad += flag != "OK"
    print(d.name.ljust(20), flag, f"worst vs FEC {worst:.3%}", {k: round(v / ref - 1, 4) for k, v in out.items() if k != "state"} if flag != "OK" else "")
sys.exit(1 if bad else 0)
