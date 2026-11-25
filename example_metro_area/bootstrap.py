"""Notebook setup. The study runs in a separate process (`worker.py`), so the notebook's own kernel never has to restart:

    import bootstrap; study = bootstrap.setup(); aoi = study.get_area("Cambridge, Massachusetts, USA", "municipality")

Where the study packages are already installed (your own machine) the worker uses this Python. Otherwise (Google Colab) they are installed once
into a private environment, next to Colab's own packages, which are left as they are.
"""
import codecs
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = Path(os.environ.get("PTLOS_ENV") or Path.home() / ".cache" / "ptlos_env")
#: import name of each line of requirements.txt, in the same order
MODULES = ["geohierarchy", "pycensus", "pyGTFSHandler", "UrbanAccessAnalyzer", "transitlos",
           "rasterio", "folium", "freestiler", "pmtiles", "anywidget", "ipywidgets", "mapclassify", "polars", "duckdb"]
#: optional API keys (more feed sources, some censuses); in Colab they are read from its Secrets, anywhere from the environment
KEYS = ("TRANSITLAND_API_KEY", "NAP_API_KEY", "DESTATIS_GENESIS_TOKEN", "MOBILITY_DATABASE_REFRESH_TOKEN")
#: what the notebook's own kernel needs for the maps (the study packages are not needed there)
KERNEL = [("anywidget", "anywidget>=0.9"), ("ipywidgets", "ipywidgets>=8"), ("matplotlib", "matplotlib>=3.7")]


def _requirements() -> list:
    return [x.strip() for x in (HERE / "requirements.txt").read_text().splitlines() if x.strip()]


def _missing() -> list:
    return [req for mod, req in zip(MODULES, _requirements()) if importlib.util.find_spec(mod) is None]


def python() -> str:
    """The Python that runs the study: this one when it has every package, else the private environment (made on first use)."""
    missing = _missing()
    if not missing:
        return sys.executable
    py = ENV / "bin" / "python"
    if not (ENV / ".ready").is_file():
        print("Installing the study packages (first time only, a few minutes; the notebook does not restart)...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = [sys.executable, "-m", "uv"]
        subprocess.check_call(uv + ["venv", "-q", "--system-site-packages", "--python", sys.executable, str(ENV)])
        subprocess.check_call(uv + ["pip", "install", "-q", "--python", str(py), *_requirements()])
        (ENV / ".ready").write_text("ok")
    return str(py)


def call(fn: str, *args, **kwargs):
    """Run `example.<fn>(*args, **kwargs)` in the study environment, printing its output live; returns its result (areas as dicts)."""
    out = tempfile.mkstemp(suffix=".json")[1]
    request = json.dumps({"fn": fn, "args": args, "kwargs": kwargs, "out": out})
    proc = subprocess.Popen([python(), str(HERE / "worker.py"), request], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    while chunk := os.read(proc.stdout.fileno(), 4096):
        sys.stdout.write(decoder.decode(chunk))
        sys.stdout.flush()
    if proc.wait():
        raise RuntimeError(f"{fn} failed (exit code {proc.returncode}); the error is in the output above.")
    text = Path(out).read_text()
    Path(out).unlink(missing_ok=True)
    return json.loads(text) if text else None


class Study:
    """`study.<function>(...)` runs that function of `example.py` (see its docstrings) in the study environment."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: call(name, *args, **kwargs)


def _colab_secrets() -> None:
    """Keys saved as Colab Secrets (the key icon in the sidebar) become environment variables, which the study process inherits."""
    try:
        from google.colab import userdata
    except ImportError:
        return
    for name in KEYS:
        if not os.environ.get(name):
            try:
                os.environ[name] = userdata.get(name)
            except Exception:  # noqa: BLE001 -- no such secret, or no access granted: the source is skipped
                pass


def setup() -> Study:
    """Make the notebook ready (the map packages; the study packages come with the first call) and return the study."""
    missing = [req for mod, req in KERNEL if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    sys.path.insert(0, str(HERE))
    try:
        from google.colab import output
    except ImportError:   # not in Colab
        pass
    else:
        output.enable_custom_widget_manager()
    _colab_secrets()
    python()
    return Study()
