"""Experiment Q: graded distribution shift and padded parameter-box guards (manuscript Section 8).

Per replicate and NMF model, ``N_SPLITS`` random calibration sets of ``T`` queries are drawn from the in-range audit pool
(``generate_features.py``) and rules are applied to the shifted pools of ``generate_shift.py`` (shift sizes ``config.SHIFTS``,
fractions of the box width beyond the box; 0 = fresh in-range queries). Indicators: ``eta`` and the cross-fitted learned
indicator (``xconf``, folds and search as in ``run_crossfit.py``). Rules: class-conditional at the levels of ``ALPHAS``.
Guards: ``none``; padded guards ``pad<g>``, which send every query farther than ``g`` box widths outside the box to the
fallback solve (``pad0`` is the strict guard of the manuscript; ``g`` in ``PADS``). Reported per shift size, indicator,
guard and level: miss rate among the silent failures of the pool, false-alarm rate, fallback share (replicates are the
independent units, as in ``run_coverage.py``).
    python scripts/run_shift.py --problem block
Writes ``results/shift_<problem>.json``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.config import (FEATURE_DIR, N_SHIFT, PROBLEMS, RESULTS, SEEDS, SHIFTS, features_path,   # noqa: E402
                                  shift_path)
from contactscreen.conformal import conformal_threshold, labels                    # noqa: E402
from contactscreen.environment import environment_info                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import summarise                                                  # noqa: E402
from run_crossfit import ALPHAS, BUDGETS, MODELS, counts, xconf_flags              # noqa: E402

log = logging.getLogger("run_shift")
N_SPLITS = 20
PADS = (0.0, 0.05, 0.10)                         # guard padding in box widths


def replicate_counts(feat, shift, key, budget: int, rng) -> dict:
    g = lambda f: feat[f"pool|{key[0]}|{key[1]}|{f}"]
    lab, X = labels(g("eta"), g("err"), g("converged")), g("X")
    s_eta, s_err, s_conv, s_X = (shift[f"{key[0]}|{key[1]}|{f}"] for f in ("eta", "err", "converged", "X"))
    s_lab = labels(s_eta, s_err, s_conv)
    dist = shift["shift"]
    totals: dict = {}
    for _ in range(N_SPLITS):
        cal = rng.permutation(len(lab.eta))[:budget]
        flags = {"eta": {}, "xconf": xconf_flags(X, lab, cal, s_X, rng)}
        for a in ALPHAS:
            tau = conformal_threshold(lab.eta[cal][lab.silent[cal]], a)
            flags["eta"][a] = (s_lab.eta >= tau, bool(np.isfinite(tau)))
        for variant, by_alpha in flags.items():
            for a, (fl, defined) in by_alpha.items():
                for guard, pad in (("none", None),) + tuple((f"pad{g_}", g_) for g_ in PADS):
                    flagged = fl if pad is None else fl | (dist > pad + 1e-12)
                    for s in SHIFTS:
                        sel = np.isclose(dist, s)
                        test = type(s_lab)(s_lab.eta[sel], s_lab.silent[sel], s_lab.good[sel], s_lab.converged[sel])
                        k = f"{variant}|{guard}|cc{a}|{s}"
                        totals[k] = totals.get(k, 0) + counts(test, flagged[sel], defined)
    return totals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    args = parser.parse_args()
    files = sorted(FEATURE_DIR.glob(f"{args.problem}_seed*.npz"))
    keys = [("nmf", r[0]) for r in MODELS[args.problem]]
    budget = BUDGETS[args.problem][-1]
    rng, boot_rng = np.random.default_rng(SEEDS["split"]), np.random.default_rng(SEEDS["bootstrap"])
    per_cell: dict = {}
    used = 0
    for path in files:
        seed = int(path.stem.split("seed")[1])
        spath = shift_path(args.problem, seed)
        if not spath.exists():
            raise FileNotFoundError(f"{spath} missing; run scripts/generate_shift.py first")
        feat, shift = np.load(path), np.load(spath)
        for key in keys:
            for k, row in replicate_counts(feat, shift, key, budget, rng).items():
                per_cell.setdefault(f"{key[0]}|{key[1]}|{k}", []).append(row)
        used += 1
        log.info("replicate %d done", seed)
    out = {"problem": args.problem, "n_replicates": used, "n_splits": N_SPLITS, "budget": budget, "pads": list(PADS),
           "shifts": list(SHIFTS), "n_per_shift": N_SHIFT[args.problem], "models": [f"{k[0]}|{k[1]}" for k in keys],
           "seeds": SEEDS, "environment": environment_info(), "cells": {k: summarise(v, boot_rng) for k, v in per_cell.items()}}
    path = RESULTS / f"shift_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    f = lambda x: " -- " if x is None else f"{x:.3f}"
    print(f"\n=== Q ({args.problem}): cc0.1, miss / fallback share by shift {list(SHIFTS)}")
    for m in out["models"]:
        for variant in ("eta", "xconf"):
            for guard in ("none", "pad0.0", "pad0.05", "pad0.1"):
                row = [out["cells"][f"{m}|{variant}|{guard}|cc0.1|{s}"] for s in SHIFTS]
                print(f"{m:<7}{variant:<6}{guard:<8}" + " ".join(f"{f(c['miss'])}/{f(c['flagged'])}" for c in row))


if __name__ == "__main__":
    main()
