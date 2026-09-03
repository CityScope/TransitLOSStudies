import sys
from pathlib import Path

# `code/` is a package (has __init__.py) inside city_science_network -- make it
# importable as `code.pipeline` regardless of which directory pytest is
# invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
