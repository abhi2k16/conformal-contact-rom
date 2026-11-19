"""Experiment X: an exactly valid cross-fitted rule ("all folds") against the CV+-style rule of experiment K.

All-folds rule. The ``T`` calibration queries are split into ``K`` folds, with the failures dealt out evenly (stratified
by label; the assignment depends on labels only). Fold ``j`` has a model ``s_j`` fitted on the queries of the other
folds and the class-conditional threshold ``tau_j`` = ``k_j``-th smallest ``s_j`` score of the failures *of fold j*,
``k_j = floor(alpha (n_j + 1))``. A query is accepted only if ``s_j(x) < tau_j`` for every ``j``. Given the training
queries of fold ``j``, ``s_j`` is a fixed function, the failures of fold ``j`` and a new failure are exchangeable, so
``P(accept via fold j | failure) <= k_j / (n_j + 1) <= alpha``; accepting requires every fold to accept, so
``P(accept | failure) <= alpha`` exactly (conditional on the labels and the fold partition). The price: every fold
needs ``n_j >= ceil(1 / alpha) - 1`` failures, so the rule is defined only for ``n >= K (ceil(1/alpha) - 1)``
failures, and flagging if any fold flags is conservative.

Same design as ``run_crossfit.py`` (equal budget, same splits for all variants, NMF models, replicate bootstrap):
variants ``eta``, ``xconf`` (random folds, as in K) and ``allfolds``.
    python scripts/run_allfolds.py --problem block
Writes ``results/allfolds_<problem>.json``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.config import FEATURE_DIR, PROBLEMS, RESULTS, SEEDS              # noqa: E402
from contactscreen.conformal import conformal_threshold, labels                     # noqa: E402
from contactscreen.environment import environment_info                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import summarise                                                  # noqa: E402
from run_crossfit import ALPHAS, BUDGETS, MODELS, N_FOLDS_XC, N_SPLITS, counts, fit, xconf_flags   # noqa: E402

log = logging.getLogger("run_allfolds")


def stratified_folds(silent: np.ndarray, n_folds: int, rng) -> np.ndarray:
    """Fold index per calibration query: failures dealt out evenly, the other queries likewise."""
    folds = np.zeros(len(silent), int)
    for mask in (silent, ~silent):
        idx = np.flatnonzero(mask)
        folds[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
    return folds


def allfolds_flags(X, lab, cal, test_X, rng) -> dict:
    """``{alpha: (flags, defined)}`` of the all-folds rule on one split (module docstring)."""
    folds = stratified_folds(lab.silent[cal], N_FOLDS_XC, rng)
    fold_fail_scores, test_scores = [], []
    for j in range(N_FOLDS_XC):
        train = cal[(folds != j) & lab.converged[cal]]
        scorer = fit(X[train], lab.silent[train], int(rng.integers(1 << 30)))
        if scorer is None:
            return {a: (np.ones(len(test_X), bool), False) for a in ALPHAS}
        in_fold = cal[(folds == j) & lab.silent[cal]]
        fold_fail_scores.append(scorer.score(X[in_fold]))
        test_scores.append(scorer.score(test_X))
    out = {}
    for a in ALPHAS:
        taus = [conformal_threshold(s, a) for s in fold_fail_scores]
        if not all(np.isfinite(t) for t in taus):
            out[a] = (np.ones(len(test_X), bool), False)
            continue
        flags = np.any([ts >= t for ts, t in zip(test_scores, taus)], axis=0)
        out[a] = (flags, True)
    return out


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
        variants["xconf"] = xconf_flags(X, lab, cal, X[test_idx], rng)
        variants["allfolds"] = allfolds_flags(X, lab, cal, X[test_idx], rng)
        for name, by_alpha in variants.items():
            for a, (flags, defined) in by_alpha.items():
                k = f"{name}|cc{a}"
                totals[k] = totals.get(k, 0) + counts(test, flags, defined)
    return totals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    args = parser.parse_args()
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
           "budgets": list(BUDGETS[args.problem]), "models": [f"{k[0]}|{k[1]}" for k in keys], "seeds": SEEDS,
           "environment": environment_info(), "cells": {k: summarise(v, boot_rng) for k, v in per_cell.items()}}
    path = RESULTS / f"allfolds_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    f = lambda x: " -- " if x is None else f"{x:.3f}"
    print(f"\n=== X ({args.problem}): cc0.1, larger budget; miss | false alarm | defined")
    T = out["budgets"][-1]
    for m in out["models"]:
        print(m, " | ".join(f"{n} {f(out['cells'][f'{m}|{T}|{n}|cc0.1']['miss'])} {f(out['cells'][f'{m}|{T}|{n}|cc0.1']['fpr'])} "
                            f"{out['cells'][f'{m}|{T}|{n}|cc0.1']['defined']:.2f}" for n in ("eta", "xconf", "allfolds")))


if __name__ == "__main__":
    main()
