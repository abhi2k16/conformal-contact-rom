"""Aggregate the per-replicate timing runs of experiment J (``run_learned_speedup.py``) over replicates.

Replicates are the independent units: for every model and every (variant, rule) cell the speedups, the fallback share
and the failure counts are averaged over replicates, with a 95 % percentile-bootstrap interval of the mean over replicates;
the miss rate is pooled over replicates (missed / silent failures, replicates where the rule was defined). The one-time
costs (training snapshots and model construction, calibration solves, fitting of the cross-fitted score) are averaged the
same way. Writes ``results/learned_speedup_<problem>.json`` in the layout of a single run (so that the paper tools read it
unchanged) with ``*_ci`` and ``n_replicates`` added.
    python scripts/aggregate_speedup.py --problem layered
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.config import PROBLEMS, RESULTS, SEEDS      # noqa: E402

log = logging.getLogger("aggregate_speedup")
N_BOOT = 2000


def mean_ci(values, rng) -> tuple:
    """Mean over replicates and its 95 % percentile-bootstrap interval."""
    x = np.asarray(values, float)
    boot = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return float(x.mean()), [float(v) for v in np.percentile(boot, [2.5, 97.5])]


def aggregate_cell(cells: list, n_silent: list, rng) -> dict:
    """One (variant, rule) cell over replicates."""
    out = {"draws": cells[0]["draws"], "defined": float(np.mean([c["defined"] for c in cells]))}
    for key in ("fallback", "speedup_cold", "speedup_warm"):
        out[key], out[f"{key}_ci"] = mean_ci([c[key] for c in cells], rng)
    weights = np.array([n * c["draws"] for c, n in zip(cells, n_silent)], float)
    miss = [(c["miss_rate"], w) for c, w in zip(cells, weights) if c["miss_rate"] is not None]
    out["miss_rate"] = float(sum(m * w for m, w in miss) / sum(w for _, w in miss)) if miss else None
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="layered")
    args = parser.parse_args()
    files = sorted(RESULTS.glob(f"learned_speedup_{args.problem}_seed*.json"))
    if not files:
        raise FileNotFoundError("no per-replicate timing runs; run scripts/run_learned_speedup.py --seed s first")
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rng = np.random.default_rng(SEEDS["bootstrap"])
    first = runs[0]
    out = {"problem": args.problem, "n_replicates": len(runs), "replicate_files": [f.name for f in files],
           "n_time": first["n_time"], "repeats": first["repeats"], "draws": first["draws"], "budget": first["budget"],
           "fit_sizes": first["fit_sizes"], "seed_rels": first["seed_rels"], "seeds": SEEDS, "environment": first["environment"],
           "models": {}}
    for name in first["models"]:
        per = [r["models"][name] for r in runs]
        n_silent = [m["n_silent"] for m in per]
        entry = {"silent_rate": float(np.mean([m["silent_rate"] for m in per])), "n_silent": int(sum(n_silent)),
                 "n_silent_per_replicate": n_silent}
        for key in ("t_cold_ms_mean", "t_warm_ms_mean", "t_rom_ms_mean", "t_feat_ms_mean"):
            entry[key], entry[f"{key}_ci"] = mean_ci([m[key] for m in per], rng)
        exact_error = {}
        for key in ("fallback", "speedup_cold", "speedup_warm"):
            exact_error[key], exact_error[f"{key}_ci"] = mean_ci([m["exact_error"][key] for m in per], rng)
        exact_error["missed"], exact_error["n_silent"] = int(sum(m["exact_error"]["missed"] for m in per)), int(sum(n_silent))
        entry["exact_error"] = exact_error
        names = sorted({c for m in per for c in m["cells"]}, key=lambda c: list(first["models"][name]["cells"]).index(c)
                       if c in first["models"][name]["cells"] else 99)
        entry["cells"] = {}
        for cell in names:                       # a variant is absent in a replicate where its indicator could not be fitted
            have = [(m["cells"][cell], n) for m, n in zip(per, n_silent) if cell in m["cells"]]
            entry["cells"][cell] = {**aggregate_cell([c for c, _ in have], [n for _, n in have], rng), "n_replicates": len(have)}
        entry["t_fit_xconf_s"], entry["t_fit_xconf_s_ci"] = mean_ci([r["one_time"]["t_fit_xconf_s"][name] for r in runs], rng)
        out["models"][name] = entry
    one = {"n_train": first["one_time"]["n_train"], "n_calibration": first["one_time"]["n_calibration"]}
    for key in ("t_build_s", "t_calibration_solves_s"):
        one[key], one[f"{key}_ci"] = mean_ci([r["one_time"][key] for r in runs], rng)
    out["one_time"] = one
    path = RESULTS / f"learned_speedup_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s from %d replicates", path, len(runs))
    for name, m in out["models"].items():
        c = m["cells"]["xconf@160|cc0.1"]
        print(f"{name}: xconf cc0.1 speedup cold {c['speedup_cold']:.2f} {c['speedup_cold_ci']} warm {c['speedup_warm']:.2f} "
              f"{c['speedup_warm_ci']} miss {c['miss_rate']}")


if __name__ == "__main__":
    main()
