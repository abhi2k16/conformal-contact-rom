"""Experiment C: do the screening rules keep their miss rate for queries outside the training range?

Rules are calibrated on random subsets of the in-range pool (as in ``run_coverage.py``) and scored on the out-of-range
pool (loads 1.3-2x the training maximum, load or foundation centres moved out of range, and for the layered block a
stiffness ratio above the training range). Exchangeability fails there, so no guarantee is expected. Remedy tested: a
parameter-space check that sends every query farther than ``margin`` (relative to the training-box width, in the
largest coordinate) from the training box straight to the full-order solve.
    python scripts/run_outside.py --problem block
Writes ``results/outside_<problem>.json``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import load_audit                                  # noqa: E402
from contactscreen.config import AUDIT_DIR, PROBLEMS, RESULTS, SEEDS, make_problem    # noqa: E402
from contactscreen.environment import environment_info                       # noqa: E402
from contactscreen.conformal import conformal_threshold, flagged, heuristic_threshold, labels, risk_threshold   # noqa: E402

log = logging.getLogger("run_outside")

N_CAL = 160
N_SPLITS = 100
MARGINS = (None, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0)    # None: no check; 0.0: every query outside the box is flagged
                                                          # distances are in units of the training-box width per coordinate (l_inf)
RULES = {"cc0.05": lambda lab: (conformal_threshold(lab.eta[lab.silent], 0.05), False),
         "cc0.1": lambda lab: (conformal_threshold(lab.eta[lab.silent], 0.10), False),
         "crc0.01": lambda lab: (risk_threshold(lab.eta, lab.silent, 0.01), False),
         "heur": lambda lab: (heuristic_threshold(lab.eta[lab.silent]), True)}


def box_distance(mus: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Largest coordinate-wise distance to the training box, in units of the box width (0 inside the box)."""
    return np.max(np.maximum(np.maximum(lo - mus, mus - hi), 0.0) / (hi - lo), axis=1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    parser.add_argument("--indicator", choices=("eta", "eta2"), default="eta", help="indicator version (eta2: problem-level scales)")
    args = parser.parse_args()
    files = sorted(AUDIT_DIR.glob(f"{args.problem}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no audits for {args.problem}; run scripts/generate_audits.py first")
    problem = make_problem(args.problem)
    audits = [load_audit(f) for f in files]
    models = [tuple(m) for m in audits[0].meta["settings"]["models"]]
    rng = np.random.default_rng(SEEDS["split"])
    out = {"problem": args.problem, "indicator": args.indicator, "n_replicates": len(audits), "n_cal": N_CAL, "n_splits": N_SPLITS,
           "margins": [m for m in MARGINS if m is not None], "seeds": SEEDS, "environment": environment_info(), "cells": {}}
    for key in models:
        acc = {(rule, m): np.zeros(4) for rule in RULES for m in MARGINS}        # miss, silent, flagged, n
        for a in audits:
            g = lambda tag, f: a.arrays[f"{tag}|{key[0]}|{key[1]}|{f}"]
            pool = labels(g("pool", args.indicator), g("pool", "err"), g("pool", "converged"))
            far = labels(g("out", args.indicator), g("out", "err"), g("out", "converged"))
            dist = box_distance(a.arrays["out|mu"], problem.lo, problem.hi)
            for _ in range(N_SPLITS):
                idx = rng.permutation(len(pool.eta))[:N_CAL]
                cal = type(pool)(pool.eta[idx], pool.silent[idx], pool.good[idx], pool.converged[idx])
                for rule, fn in RULES.items():
                    tau, strict = fn(cal)
                    fl = flagged(far, tau, strict)
                    for m in MARGINS:
                        fm = fl if m is None else fl | (dist > m)
                        acc[(rule, m)] += [(far.silent & ~fm).sum(), far.silent.sum(), fm.sum(), len(fm)]
        for (rule, m), v in acc.items():
            out["cells"][f"{key[0]}|{key[1]}|{rule}|" + ("none" if m is None else str(m))] = {
                "miss": float(v[0] / v[1]) if v[1] else None, "flagged": float(v[2] / v[3]), "n_silent": int(v[1] / N_SPLITS / len(audits))}
        log.info("model %s done", key)
    path = RESULTS / (f"outside_{args.problem}.json" if args.indicator == "eta" else f"outside_{args.problem}_{args.indicator}.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    print_table(out, models, args.problem)


def print_table(out: dict, models: list, problem: str) -> None:
    fmt = lambda x: "  --" if x is None else f"{x:.3f}"
    name = lambda m: "none" if m is None else str(m)
    print(f"\n=== {problem}: out-of-range queries, calibration {N_CAL} in-range queries; miss rate among silent failures | fallback share")
    for key in models:
        for rule in ("cc0.05", "crc0.01", "heur"):
            parts = []
            for m in MARGINS:
                cell = out["cells"][f"{key[0]}|{key[1]}|{rule}|{name(m)}"]
                parts.append(f"{name(m):>5}: {fmt(cell['miss'])}|{fmt(cell['flagged'])}")
            print(f"  {key[0]:<4}{key[1]:>3} {rule:<8} " + " ".join(parts))


if __name__ == "__main__":
    main()
