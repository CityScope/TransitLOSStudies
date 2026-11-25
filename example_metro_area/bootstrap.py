"""Notebook setup that runs before anything else is imported: installs the missing packages and puts the study code on the path.

    import bootstrap; IN_COLAB = bootstrap.setup()
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: import name of each line of requirements.txt, in the same order
MODULES = ["geohierarchy", "pycensus", "pyGTFSHandler", "UrbanAccessAnalyzer", "transitlos",
           "rasterio", "folium", "freestiler", "pmtiles", "ipyleaflet", "ipywidgets", "mapclassify"]


def setup() -> bool:
    """Install what is missing (Colab then restarts the session: run all cells again) and enable widgets. True when running in Colab."""
    in_colab = importlib.util.find_spec("google.colab") is not None
    lines = [x.strip() for x in (HERE / "requirements.txt").read_text().splitlines() if x.strip()]
    missing = [req for mod, req in zip(MODULES, lines) if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing:", ", ".join(m.split(" @ ")[0] for m in missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        if in_colab:
            print("Installed. The session restarts now; run all cells again (Runtime -> Run all).")
            os.kill(os.getpid(), 9)
        raise SystemExit("Installed. Restart the kernel, then run all cells again.")
    sys.path.insert(0, str(HERE))
    if in_colab:
        from google.colab import output
        output.enable_custom_widget_manager()
    return in_colab
