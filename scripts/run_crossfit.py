"""Experiment K: cross-conformal fitting of the learned score (no separate fitting set).

For every replicate, NMF model and budget ``T`` of full-order solves, ``N_SPLITS`` random splits of the audit pool give
``T`` calibration queries and a test part (the rest). Three variants spend the same ``T`` solves:
  * ``eta``    the standard indicator, class-conditional split conformal on all ``T``,
  * ``split``  learned score fitted on ``T/2`` of them, split conformal on the other ``T/2`` (Section 12 design),
  * ``xconf``  learned score cross-fitted in ``N_FOLDS_XC`` folds on all ``T``; each calibration failure is scored by the
               fold model that did not see it (``conformal.cross_conformal_flags``; theory: miss <= 2 alpha, check alpha).
A learned variant whose score cannot be fitted (too few failures or good answers in a training part) flags every query.
Both learned variants use the same reduced hyper-parameter search (``FAST_GRID``) to keep the run short.
    python scripts/run_crossfit.py --problem block
Writes ``results/crossfit_<problem>.json``. Needs ``generate_features.py`` output (the ``pool`` part only).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.config import FEATURE_DIR, PROBLEMS, RESULTS, SEEDS                # noqa: E402
from contactscreen.conformal import conformal_threshold, cross_conformal_flags, labels, min_failures   # noqa: E402
from contactscreen.environment import environment_info                                 # noqa: E402
from contactscreen.features import FeatureScorer                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import summarise                                                     # noqa: E402

log = logging.getLogger("run_crossfit")

ALPHAS = (0.05, 0.10)
N_SPLITS = 20
N_FOLDS_XC = 5
FAST_GRID = {"ridges": (1.0, 10.0), "n_folds": 3}
BUDGETS = {"membrane": (200, 400), "block": (120, 240), "layered": (80, 160)}
MODELS = {"membrane": ((24,), (36,), (48,)), "block": ((8,), (12,), (16,), (24,)), "layered": ((8,), (12,), (16,), (24,))}


def counts(test, flags: np.ndarray, defined: bool) -> np.ndarray:
    """Row in ``run_coverage.COLS`` order; non-converged reduced answers are always flagged."""
    fl = flags | ~test.converged
    return np.array([(test.silent & ~fl).sum(), test.silent.sum(), (test.good & fl).sum(), test.good.sum(),
                     fl.sum(), len(fl), float(defined), 1.0], float)


def fit(X, silent, seed):
    return FeatureScorer.fit(X, silent, seed=seed, **FAST_GRID)


def split_flags(X, lab, cal, test_X, rng) -> dict:
    """Learned score fitted on the first half of ``cal``, split conformal on the second half."""
    fit_rows, cal_rows = cal[: len(cal) // 2], cal[len(cal) // 2:]
    rows = fit_rows[lab.converged[fit_rows]]
    scorer = fit(X[rows], lab.silent[rows], int(rng.integers(1 << 30)))
    if scorer is None:
        return {a: (np.ones(len(test_X), bool), False) for a in ALPHAS}
    s_cal, s_test = scorer.score(X[cal_rows]), scorer.score(test_X)
    out = {}
    for a in ALPHAS:
        tau = conformal_threshold(s_cal[lab.silent[cal_rows]], a)
        out[a] = (s_test >= tau, bool(np.isfinite(tau)))
    return out


def xconf_flags(X, lab, cal, test_X, rng) -> dict:
    """Learned score cross-fitted in ``N_FOLDS_XC`` folds on ``cal``; cross-conformal class-conditional rule."""
    folds = rng.permutation(len(cal)) % N_FOLDS_XC
    oof, test_scores = np.zeros(len(cal)), np.zeros((N_FOLDS_XC, len(test_X)))
    for k in range(N_FOLDS_XC):
        train = cal[(folds != k) & lab.converged[cal]]
        scorer = fit(X[train], lab.silent[train], int(rng.integers(1 << 30)))
        if scorer is None:
            return {a: (np.ones(len(test_X), bool), False) for a in ALPHAS}
        oof[folds == k] = scorer.score(X[cal[folds == k]])
        test_scores[k] = scorer.score(test_X)
    is_fail = lab.silent[cal]
    return {a: cross_conformal_flags(oof[is_fail], folds[is_fail], test_scores, a) for a in ALPHAS}


def replicate_counts(data, key, budget: int, rng) -> dict:
    g = lambda f: data[f"pool|{key[0]}|{key[1]}|{f}"]
    lab, X = labels(g("eta"), g("err"), g("converged")), g("X")
    totals: dict = {}
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(lab.eta))
        cal, test_idx = perm[:budget], perm[budget:]
        test = type(lab)(lab.eta[test_idx], lab.silent[test_idx], lab.good[test_idx], lab.converged[test_idx])
        variants = {"eta": {}}
        for a in ALPHAS:
            tau = conformal_threshold(lab.eta[cal][lab.silent[cal]], a)
            variants["eta"][a] = (test.eta >= tau, bool(np.isfinite(tau)))
        variants["split"] = split_flags(X, lab, cal, X[test_idx], rng)
        variants["xconf"] = xconf_flags(X, lab, cal, X[test_idx], rng)
        for name, by_alpha in variants.items():
            for a, (flags, defined) in by_alpha.items():
                k = f"{name}|cc{a}"
                totals[k] = totals.get(k, 0) + counts(test, flags, defined)
    return totals


def print_tables(out: dict) -> None:
    f = lambda x: " -- " if x is None else f"{x:5.3f}"
    print(f"\n=== K ({out['problem']}): {out['n_replicates']} replicates x {out['n_splits']} splits; miss | false alarm | defined")
    for model in out["models"]:
        for budget in out["budgets"]:
            line = f"{model:<7} T={budget:<4}"
            for v in ("eta", "split", "xconf"):
                for a in ALPHAS:
                    c = out["cells"][f"{model}|{budget}|{v}|cc{a}"]
                    line += f" | {v} cc{a}: {f(c['miss'])} {f(c['fpr'])} {f(c['defined'])}"
            print(line)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    args = p.parse_args()
    files = sorted(FEATURE_DIR.glob(f"{args.problem}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no feature files for {args.problem}; run scripts/generate_features.py first")
    keys = [("nmf", r[0]) for r in MODELS[args.problem]]
    rng, boot_rng = np.random.default_rng(SEEDS["split"]), np.random.default_rng(SEEDS["bootstrap"])
    per_cell: dict = {}
    for path in files:
        data = np.load(path)
        for key in keys:
            for budget in BUDGETS[args.problem]:
                for k, row in replicate_counts(data, key, budget, rng).items():
                    per_cell.setdefault(f"{key[0]}|{key[1]}|{budget}|{k}", []).append(row)
        log.info("%s processed", path.name)
    out = {"problem": args.problem, "n_replicates": len(files), "n_splits": N_SPLITS, "n_folds": N_FOLDS_XC,
           "fast_grid": FAST_GRID, "budgets": list(BUDGETS[args.problem]), "models": [f"{k[0]}|{k[1]}" for k in keys],
           "min_failures": {str(a): min_failures(a) for a in ALPHAS}, "seeds": SEEDS, "environment": environment_info(),
           "cells": {k: summarise(rows, boot_rng) for k, rows in per_cell.items()}}
    path = RESULTS / f"crossfit_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    print_tables(out)


if __name__ == "__main__":
    main()
