"""Experiment I: does a learned screening score beat the hand-made indicator ``eta`` under the same conformal rules?

For every replicate and reduced model, a ``FeatureScorer`` is fitted on ``n_fit`` fresh queries with full-order solves
(``generate_features.py``, tag ``fit``), then scored on the audit's in-range pool and out-of-range pool. Rules are
calibrated on random ``n_cal``-query subsets of the pool exactly as in ``run_coverage.py`` (the same random splits for
every variant, so differences are not split noise). Every variant spends the same budget ``T`` of full-order solves:
  * ``eta@T``            the standard indicator, all ``T`` queries used for calibration,
  * ``learned@T:F``      the learned score, ``F`` queries used to fit it and ``T - F`` to calibrate (``F`` = T/4, T/2).
Reported: AUC, and for ``cc`` (alpha 0.05, 0.10) and ``crc`` (delta 0.01) the miss rate, false-alarm rate and the share of
splits where the rule is defined; in range and out of range (calibrate in range, test out of range).
    python scripts/run_learned.py --problem membrane
Writes ``results/learned_<problem>.json``. Needs ``generate_audits.py`` and ``generate_features.py`` output.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import load_audit                                            # noqa: E402
from contactscreen.config import AUDIT_DIR, FEATURE_DIR, PROBLEMS, RESULTS, SEEDS       # noqa: E402
from contactscreen.conformal import auc, conformal_threshold, labels, outcome, risk_threshold   # noqa: E402
from contactscreen.environment import environment_info                                 # noqa: E402
from contactscreen.features import FeatureScorer                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import COLS, N_SPLITS, subset, summarise                             # noqa: E402

log = logging.getLogger("run_learned")

ALPHAS, DELTA = (0.05, 0.10), 0.01
BUDGETS = {"membrane": (200, 400), "block": (120, 240), "layered": (80, 160)}      # full-order solves for fitting + calibration
FIT_SHARES = (4, 2)                                                                # fit on T / share queries


def thresholds(lab) -> dict:
    s = lab.eta[lab.silent]
    out = {f"cc{a}": conformal_threshold(s, a) for a in ALPHAS}
    out[f"crc{DELTA}"] = risk_threshold(lab.eta, lab.silent, DELTA)
    return out


def counts_row(test, tau) -> np.ndarray:
    o = outcome(test, tau)
    return np.array([o["miss"], o["silent"], o["false_alarm"], o["good"], o["flagged"], o["n"], float(o["tau_finite"]), 1.0])


def replicate_counts(pool_scores: dict, out_scores: dict, pool_truth, out_truth, rng) -> dict:
    """``{variant|setting|rule: counts}`` summed over ``N_SPLITS`` splits; ``pool_scores[v] = (score, n_cal_v)``."""
    totals: dict = {}
    n_pool = len(pool_truth[0])
    for _ in range(N_SPLITS):
        perm = rng.permutation(n_pool)
        for name, (score, n_v) in pool_scores.items():
            lab = labels(score, *pool_truth)
            cal, test = subset(lab, perm[:n_v]), subset(lab, perm[n_v:])
            out_lab = labels(out_scores[name], *out_truth)
            for rule, tau in thresholds(cal).items():
                for setting, target in (("in", test), ("out", out_lab)):
                    key = f"{name}|{setting}|{rule}"
                    totals[key] = totals.get(key, 0) + counts_row(target, tau)
    return totals


def process_replicate(path: Path, problem: str, models: list, rng) -> tuple:
    """Per-model, per-variant counts and AUCs of one replicate (a variant is absent if no score could be fitted)."""
    data = np.load(path)
    results, aucs = {}, {}
    for kind, r in models:
        g = lambda tag, f: data[f"{tag}|{kind}|{r}|{f}"]
        fit_lab = labels(g("fit", "eta"), g("fit", "err"), g("fit", "converged"))
        pool_truth, out_truth = (g("pool", "err"), g("pool", "converged")), (g("out", "err"), g("out", "converged"))
        pool_scores, out_scores = {}, {}
        for budget in BUDGETS[problem]:
            pool_scores[f"eta@{budget}"], out_scores[f"eta@{budget}"] = (g("pool", "eta"), budget), g("out", "eta")
            for share in FIT_SHARES:
                n_fit = budget // share
                rows = np.flatnonzero(fit_lab.converged[:n_fit])
                scorer = FeatureScorer.fit(g("fit", "X")[:n_fit][rows], fit_lab.silent[:n_fit][rows])
                if scorer is None:
                    continue
                name = f"learned@{budget}:{n_fit}"
                pool_scores[name], out_scores[name] = (scorer.score(g("pool", "X")), budget - n_fit), scorer.score(g("out", "X"))
        for name, (score, _) in pool_scores.items():
            lab = labels(score, *pool_truth)
            value = auc(lab.eta[lab.silent], lab.eta[lab.good])
            if value is not None:
                aucs.setdefault(f"{kind}|{r}|{name}", []).append(value)
        for key, row in replicate_counts(pool_scores, out_scores, pool_truth, out_truth, rng).items():
            results.setdefault(f"{kind}|{r}|{key}", []).append(row)
    return results, aucs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    args = p.parse_args()
    files = sorted(FEATURE_DIR.glob(f"{args.problem}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no feature files for {args.problem}; run scripts/generate_features.py first")
    audit0 = load_audit(AUDIT_DIR / f"{args.problem}_seed0.npz")
    models = [tuple(m) for m in audit0.meta["settings"]["models"]]
    rng, boot_rng = np.random.default_rng(SEEDS["split"]), np.random.default_rng(SEEDS["bootstrap"])
    cells: dict = {}
    auc_all: dict = {}
    for path in files:
        res, aucs = process_replicate(path, args.problem, models, rng)
        for k, rows in res.items():
            cells.setdefault(k, []).extend(rows)
        for k, v in aucs.items():
            auc_all.setdefault(k, []).extend(v)
        log.info("%s processed", path.name)
    out = {"problem": args.problem, "n_replicates": len(files), "budgets": list(BUDGETS[args.problem]), "fit_shares": list(FIT_SHARES),
           "n_splits": N_SPLITS, "seeds": SEEDS, "environment": environment_info(),
           "auc": {k: float(np.mean(v)) for k, v in auc_all.items()}, "n_replicates_present": {k: len(v) for k, v in auc_all.items()},
           "cells": {k: summarise(rows, boot_rng) for k, rows in cells.items()}}
    path = RESULTS / f"learned_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    print_tables(out, models)


def print_tables(out: dict, models: list) -> None:
    f = lambda x: " -- " if x is None else f"{x:5.3f}"
    print(f"\n=== {out['problem']}: {out['n_replicates']} replicates; T = full-order solves for fitting + calibration; cc0.1 (miss | false alarm | defined)")
    for kind, r in models:
        print(f"-- {kind} {r}")
        for budget in out["budgets"]:
            for v in [f"eta@{budget}"] + [f"learned@{budget}:{budget // s}" for s in out["fit_shares"]]:
                a, n = out["auc"].get(f"{kind}|{r}|{v}"), out["n_replicates_present"].get(f"{kind}|{r}|{v}", 0)
                line = f"   {v:<18} AUC {'  -- ' if a is None else format(a, '.3f')} ({n} rep) |"
                for setting in ("in", "out"):
                    c = out["cells"].get(f"{kind}|{r}|{v}|{setting}|cc0.1")
                    line += f" {setting}: {f(c['miss'] if c else None)} {f(c['fpr'] if c else None)} {f(c['defined'] if c else None)} |"
                print(line)


if __name__ == "__main__":
    main()
