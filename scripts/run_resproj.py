"""Experiment S: residual-projection features for the cross-fitted learned indicator (manuscript Section 3.3).

Same design as experiment K (``run_crossfit.py``): per replicate, NMF model and budget ``T``, ``N_SPLITS`` random splits
of the in-range pool into ``T`` calibration queries and a test part, the same split for every variant:
  * ``eta``       analytical indicator, class-conditional rule on all ``T``;
  * ``base``      cross-fitted learned indicator with the scalar features of Section 3.2 (as in experiment K);
  * ``full<k>``   ``base`` plus the coordinates of the residual in the first ``k`` training-residual directions;
  * ``contact<k>`` ``base`` plus the coordinates of the residual restricted to the constrained unknowns.
The projection bases come from residuals at the training parameters (``generate_residuals.py``), independent of the
calibration and test queries.
    python scripts/run_resproj.py --problem block
Writes ``results/resproj_<problem>.json``. Needs ``generate_features.py`` and ``generate_residuals.py`` output.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.config import PROBLEMS, RESULTS, SEEDS, features_path, residuals_path   # noqa: E402
from contactscreen.conformal import conformal_threshold, labels                           # noqa: E402
from contactscreen.environment import environment_info                                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import summarise                                                         # noqa: E402
from run_crossfit import ALPHAS, BUDGETS, MODELS, N_SPLITS, counts, xconf_flags           # noqa: E402

log = logging.getLogger("run_resproj")
K_VALUES = (5, 10)


def variants(feat, res, key) -> dict:
    """Feature matrices of the pool for every learned variant."""
    base = feat[f"pool|{key[0]}|{key[1]}|X"]
    pre = f"pool|{key[0]}|{key[1]}|"
    out = {"base": base}
    for k in K_VALUES:
        out[f"full{k}"] = np.column_stack([base, res[pre + "cfull"][:, :k]])
        out[f"contact{k}"] = np.column_stack([base, res[pre + "ccontact"][:, :k]])
    return out


def replicate_counts(feat, res, key, budget: int, rng) -> dict:
    g = lambda f: feat[f"pool|{key[0]}|{key[1]}|{f}"]
    lab = labels(g("eta"), g("err"), g("converged"))
    if not np.allclose(res[f"pool|{key[0]}|{key[1]}|eta"], lab.eta, rtol=1e-8, atol=1e-12):
        raise RuntimeError(f"{key}: feature and residual files disagree on eta")
    mats = variants(feat, res, key)
    totals: dict = {}
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(lab.eta))
        cal, test_idx = perm[:budget], perm[budget:]
        test = type(lab)(lab.eta[test_idx], lab.silent[test_idx], lab.good[test_idx], lab.converged[test_idx])
        flags = {"eta": {}}
        for a in ALPHAS:
            tau = conformal_threshold(lab.eta[cal][lab.silent[cal]], a)
            flags["eta"][a] = (test.eta >= tau, bool(np.isfinite(tau)))
        for name, X in mats.items():
            flags[name] = xconf_flags(X, lab, cal, X[test_idx], rng)
        for name, by_alpha in flags.items():
            for a, (fl, defined) in by_alpha.items():
                k = f"{name}|cc{a}"
                totals[k] = totals.get(k, 0) + counts(test, fl, defined)
    return totals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    args = parser.parse_args()
    seeds = sorted(int(p.stem.split("seed")[1]) for p in features_path(args.problem, 0).parent.glob(f"{args.problem}_seed*.npz"))
    if not seeds:
        raise FileNotFoundError(f"no feature files for {args.problem}")
    keys = [("nmf", r[0]) for r in MODELS[args.problem]]
    rng, boot_rng = np.random.default_rng(SEEDS["split"]), np.random.default_rng(SEEDS["bootstrap"])
    per_cell: dict = {}
    for seed in seeds:
        rpath = residuals_path(args.problem, seed)
        if not rpath.exists():
            raise FileNotFoundError(f"{rpath} missing; run scripts/generate_residuals.py first")
        feat, res = np.load(features_path(args.problem, seed)), np.load(rpath)
        for key in keys:
            for budget in BUDGETS[args.problem]:
                for k, row in replicate_counts(feat, res, key, budget, rng).items():
                    per_cell.setdefault(f"{key[0]}|{key[1]}|{budget}|{k}", []).append(row)
        log.info("replicate %d done", seed)
    out = {"problem": args.problem, "n_replicates": len(seeds), "n_splits": N_SPLITS, "k_values": list(K_VALUES),
           "budgets": list(BUDGETS[args.problem]), "models": [f"{k[0]}|{k[1]}" for k in keys], "seeds": SEEDS,
           "environment": environment_info(), "cells": {k: summarise(v, boot_rng) for k, v in per_cell.items()}}
    path = RESULTS / f"resproj_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    f = lambda x: " -- " if x is None else f"{x:.3f}"
    names = ["eta", "base"] + [f"{v}{k}" for k in K_VALUES for v in ("full", "contact")]
    print(f"\n=== S ({args.problem}): cc0.1, larger budget; miss | false alarm | defined")
    T = out["budgets"][-1]
    for m in out["models"]:
        print(m, " | ".join(f"{n} {f(out['cells'][f'{m}|{T}|{n}|cc0.1']['miss'])} {f(out['cells'][f'{m}|{T}|{n}|cc0.1']['fpr'])} "
                            f"{out['cells'][f'{m}|{T}|{n}|cc0.1']['defined']:.2f}" for n in names))


if __name__ == "__main__":
    main()
