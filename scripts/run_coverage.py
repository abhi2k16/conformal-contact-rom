"""Experiments A and B: coverage, false alarms and calibration size of the screening rules, from saved audits.

For every replicate (independent training set and query pool), reduced model and calibration size ``n_cal``, the
in-range pool is split at random into a calibration part (``n_cal`` queries) and a test part (the rest); each rule is
calibrated on the first and scored on the second. Counts are pooled over splits within a replicate; intervals are
95% percentile bootstrap intervals over replicates (the replicates are the independent units).

Rules: ``cc<alpha>`` class-conditional conformal (miss rate among silent failures <= alpha), ``crc<delta>`` conformal
risk control (silent failure that is accepted <= delta per query), ``heur`` the 0.9 x min margin rule (no guarantee).
    python scripts/run_coverage.py --problem membrane
Writes ``results/coverage_<problem>.json``.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import load_audit                                           # noqa: E402
from contactscreen.config import AUDIT_DIR, PROBLEMS, RESULTS, SEEDS               # noqa: E402
from contactscreen.conformal import (auc, conformal_threshold, heuristic_threshold, labels, min_failures,   # noqa: E402
                                     outcome, risk_threshold)
from contactscreen.environment import environment_info                              # noqa: E402

log = logging.getLogger("run_coverage")

ALPHAS = (0.05, 0.10)
DELTAS = (0.01, 0.02)
N_SPLITS = 100
N_BOOT = 2000
CI = (2.5, 97.5)
COLS = ("miss", "silent", "false_alarm", "good", "flagged", "n", "finite", "splits")


def n_cal_grid(n_pool: int) -> tuple:
    return tuple(n for n in (20, 40, 80, 160, 320) if n <= n_pool // 2)


def rule_thresholds(lab) -> dict:
    """``{rule name: (tau, strict)}`` calibrated on the labelled calibration queries ``lab``."""
    silent_scores = lab.eta[lab.silent]
    out = {f"cc{a}": (conformal_threshold(silent_scores, a), False) for a in ALPHAS}
    out.update({f"crc{d}": (risk_threshold(lab.eta, lab.silent, d), False) for d in DELTAS})
    out["heur"] = (heuristic_threshold(silent_scores), True)
    return out


def subset(lab, idx):
    return type(lab)(lab.eta[idx], lab.silent[idx], lab.good[idx], lab.converged[idx])


def replicate_counts(audit, key: tuple, n_cal: int, rng: np.random.Generator, field: str) -> dict:
    """Counts per rule, summed over ``N_SPLITS`` random calibration/test splits of the in-range pool."""
    g = lambda f: audit.arrays[f"pool|{key[0]}|{key[1]}|{f}"]
    lab = labels(g(field), g("err"), g("converged"))
    totals: dict = {}
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(lab.eta))
        cal, test = subset(lab, perm[:n_cal]), subset(lab, perm[n_cal:])
        for rule, (tau, strict) in rule_thresholds(cal).items():
            o = outcome(test, tau, strict)
            row = np.array([o["miss"], o["silent"], o["false_alarm"], o["good"], o["flagged"], o["n"], float(o["tau_finite"]), 1.0])
            totals[rule] = totals.get(rule, 0) + row
    return totals


def ratio(arr: np.ndarray, num: str, den: str):
    d = arr[:, COLS.index(den)].sum()
    return float(arr[:, COLS.index(num)].sum() / d) if d else None


def boot_ci(arr: np.ndarray, num: str, den: str, rng: np.random.Generator):
    """Percentile bootstrap interval of sum(num)/sum(den) over replicates (rows of ``arr``)."""
    ni, di = COLS.index(num), COLS.index(den)
    if arr[:, di].sum() == 0:
        return None
    draws = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    n, d = arr[draws, ni].sum(axis=1), arr[draws, di].sum(axis=1)
    ok = d > 0
    return [float(x) for x in np.percentile(n[ok] / d[ok], CI)]


def summarise(per_replicate: list, boot_rng) -> dict:
    arr = np.array(per_replicate)                          # (n_replicates, len(COLS))
    return {"miss": ratio(arr, "miss", "silent"), "miss_ci": boot_ci(arr, "miss", "silent", boot_rng),
            "risk": ratio(arr, "miss", "n"), "risk_ci": boot_ci(arr, "miss", "n", boot_rng),
            "fpr": ratio(arr, "false_alarm", "good"), "fpr_ci": boot_ci(arr, "false_alarm", "good", boot_rng),
            "flagged": ratio(arr, "flagged", "n"), "defined": ratio(arr, "finite", "splits"),
            "n_silent_test": int(arr[:, COLS.index("silent")].sum())}


def failure_rates(audits: list, key: tuple, boot_rng, field: str) -> dict:
    """Experiment B: silent-failure rate of a model, the calibration size it implies, and the discrimination (AUC) of the indicator."""
    counts, aucs = [], []
    for a in audits:
        lab = labels(*(a.arrays[f"pool|{key[0]}|{key[1]}|{f}"] for f in (field, "err", "converged")))
        counts.append([lab.silent.sum(), len(lab.eta)])
        value = auc(lab.eta[lab.silent], lab.eta[lab.good])
        if value is not None:
            aucs.append(value)
    arr = np.array(counts, float)
    p = arr[:, 0].sum() / arr[:, 1].sum()
    draws = boot_rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    ps = arr[draws, 0].sum(axis=1) / arr[draws, 1].sum(axis=1)
    lo, hi = (float(x) for x in np.percentile(ps, CI))
    need = lambda rate, a: math.ceil(min_failures(a) / rate) if rate > 0 else None
    auc_ci = [float(x) for x in np.percentile(boot_rng.choice(aucs, size=(N_BOOT, len(aucs))).mean(axis=1), CI)] if aucs else None
    return {"p": float(p), "p_ci": [lo, hi], "auc": float(np.mean(aucs)) if aucs else None, "auc_ci": auc_ci,
            **{f"queries_needed@{a}": need(p, a) for a in ALPHAS},
            **{f"queries_needed_range@{a}": [need(hi, a), need(lo, a)] for a in ALPHAS}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    parser.add_argument("--indicator", choices=("eta", "eta2"), default="eta", help="indicator version (eta2: problem-level scales)")
    args = parser.parse_args()
    files = sorted(AUDIT_DIR.glob(f"{args.problem}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no audits for {args.problem} in {AUDIT_DIR}; run scripts/generate_audits.py first")
    audits = [load_audit(f) for f in files]
    models = [tuple(m) for m in audits[0].meta["settings"]["models"]]
    n_pool = len(audits[0].arrays["pool|mu"])
    rng, boot_rng = np.random.default_rng(SEEDS["split"]), np.random.default_rng(SEEDS["bootstrap"])
    out = {"problem": args.problem, "indicator": args.indicator, "n_replicates": len(audits), "n_pool": n_pool, "n_splits": N_SPLITS,
           "n_boot": N_BOOT, "seeds": SEEDS, "environment": environment_info(), "cells": {}, "rates": {}}
    for key in models:
        out["rates"][f"{key[0]}|{key[1]}"] = failure_rates(audits, key, boot_rng, args.indicator)
        for n_cal in n_cal_grid(n_pool):
            per_rule: dict = {}
            for a in audits:
                for rule, row in replicate_counts(a, key, n_cal, rng, args.indicator).items():
                    per_rule.setdefault(rule, []).append(row)
            for rule, rows in per_rule.items():
                out["cells"][f"{key[0]}|{key[1]}|{n_cal}|{rule}"] = summarise(rows, boot_rng)
        log.info("model %s done", key)
    path = RESULTS / (f"coverage_{args.problem}.json" if args.indicator == "eta" else f"coverage_{args.problem}_{args.indicator}.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    print_tables(out, models)


def print_tables(out: dict, models: list) -> None:
    f = lambda x, w=6: " " * (w - 2) + "--" if x is None else f"{x:{w}.3f}"
    print(f"\n=== {out['problem']} ({out['indicator']}): {out['n_replicates']} replicates, pool {out['n_pool']}")
    print("B: silent-failure rate p [95% CI] and calibration queries needed (alpha 0.05 | 0.10)")
    for key in models:
        r = out["rates"][f"{key[0]}|{key[1]}"]
        print(f"  {key[0]:<4}{key[1]:>3}  p {r['p']:.3f} [{r['p_ci'][0]:.3f},{r['p_ci'][1]:.3f}]   need {r['queries_needed@0.05']} | {r['queries_needed@0.1']}   AUC {'--' if r['auc'] is None else format(r['auc'], '.3f')}")
    n_show = max(n for n in n_cal_grid(out["n_pool"]) if n <= 160)
    print(f"A: n_cal = {n_show}: miss rate [CI] | false-alarm rate | defined share")
    for key in models:
        for rule in [f"cc{a}" for a in ALPHAS] + [f"crc{d}" for d in DELTAS] + ["heur"]:
            c = out["cells"][f"{key[0]}|{key[1]}|{n_show}|{rule}"]
            ci = "--" if c["miss_ci"] is None else f"[{c['miss_ci'][0]:.3f},{c['miss_ci'][1]:.3f}]"
            print(f"  {key[0]:<4}{key[1]:>3} {rule:<8} miss {f(c['miss'])} {ci:<15} risk {f(c['risk'])}  fpr {f(c['fpr'])}  flagged {f(c['flagged'])}  defined {c['defined']:.2f}")


if __name__ == "__main__":
    main()
