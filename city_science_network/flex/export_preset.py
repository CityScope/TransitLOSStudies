"""The scoring preset of the city science score, for the flex maps: the curves of a shared fit as the map's tables.

    $TORCH_PYTHON flex/export_preset.py <shared fit name> [out.json]     # e.g. large -> flex/presets/city_score.json

Reads the saved model `results/v2_shared_<name>.pt` and the P95 of the cells' scores in `results/v2_rank_<name>.json`, and writes it with the library
(`transitlos.flex.fit.preset`): stop value M(mode) * S(speed) * H(headway) (product), the curves as tables, walking over the map's table, and the output
transform min(1, log(1 + v / eps) / P95). The fit clips item speeds per mode (bus 5-70, tram 5-60, rail 10-130 km/h); the map does not.
"""
import json
import sys
from pathlib import Path

from transitlos.flex import fit as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_cities as FC  # noqa: E402


def preset(name: str) -> dict:
    m = FC.load_shared(name)
    p95 = json.loads((HERE / "results" / f"v2_rank_{name}.json").read_text())["p95"]
    pool = json.loads((HERE / "results" / f"v2_shared_{name}.json").read_text())["pool"]
    e = tf.preset(m, "City science score 2026-11: curves shared by the cities (" + ", ".join(c.replace("_", " ").title() for c in pool) + ")", p95)
    e["run"] = f"v2_shared_{name}"
    return e


if __name__ == "__main__":
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "presets" / "city_score.json"
    out.write_text(json.dumps([preset(sys.argv[1])]))
    print("wrote", out)
