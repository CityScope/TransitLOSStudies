"""The USA study's own code. Code it shares with other studies is in `TransitLOSStudies/tools/studytools`."""

import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
