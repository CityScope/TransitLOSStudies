import sys
from pathlib import Path

# TransitLOSStudies/tools (the shared `studytools` package), from any working directory
_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import copy

import pytest


@pytest.fixture(autouse=True)
def _restore_pipeline_column_sets():
    """The pipeline adds the columns a run finds to its module-level keep-sets (`*_KEEP_COLUMNS`, `_RATE_COLUMNS`): restore
    them after every test so one test's run cannot leak into another's checks."""
    mod = sys.modules.get("studytools.pipeline")
    if mod is None:
        yield
        return
    saved = {k: copy.copy(v) for k, v in vars(mod).items() if (k.endswith("_KEEP_COLUMNS") or k == "_RATE_COLUMNS") and isinstance(v, (set, dict))}
    yield
    for k, v in saved.items():
        cur = getattr(mod, k)
        cur.clear()
        cur.update(v)
