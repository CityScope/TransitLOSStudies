"""Data locations of the fit scripts. FLEX_VARIANT=buf -> the stop-buffer flex map (../web_buf, data_buf/); unset -> the original (../web, data/)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANT = os.environ.get("FLEX_VARIANT", "")
BUF = VARIANT == "buf"
WEB = HERE.parent / ("web_buf" if BUF else "web")
DATA = HERE / ("data_buf" if BUF else "data")
FLEX_DATA = HERE.parent / ("data_buf" if BUF else "data")   # per-state flex outputs (cells.parquet for the buffer variant)
