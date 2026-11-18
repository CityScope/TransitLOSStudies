"""The city study's own code (its cities, feeds, combined map). The pipeline it runs is shared: `TransitLOSStudies/tools/studytools`."""

import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
