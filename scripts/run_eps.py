"""Sensitivity to the error tolerance: does the picture depend on defining a silent failure by eps = 0.05?

The audits store the energy error of every query, so the labels for other tolerances need no new solves. For every
tolerance in ``EPS_VALUES`` (the paper's 0.05 and a tighter and a looser one), replicate and NMF model, on the in-range
audit pool: the silent-failure rate ``p`` (replicate-bootstrap interval), the area under the ROC curve of ``eta``, the
calibration size ``ceil(n_min / p)`` for the class-conditional rule, and the class-conditional rule's miss rate,
false-alarm rate and defined share on ``N_SPLITS`` random calibration/test splits with the calibration size of
experiment A (``N_CAL``). Replicates are the independent units.
    python scripts/run_eps.py --problem block
Writes ``results/eps_<problem>.json``.
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
from contactscreen.conformal import auc, conformal_threshold, labels, min_failures, outcome   # noqa: E402
from contactscreen.environment import environment_info                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import ALPHAS, N_BOOT, summarise                                  # noqa: E402

log = logging.getLogger("run_eps")
EPS_VALUES = (0.02, 0.05, 0.10)
N_CAL = {"membrane": 160, "block": 160, "layered": 80}          # as experiment A
N_SPLITS = 100


def split_rows(lab, n_cal: int, rng) -> dict:
    """Counts of the class-conditional rule at every level, summed over ``N_SPLITS`` splits (``run_coverage.COLS`` order)."""
    totals: dict = {}
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(lab.eta))
        cal = type(lab)(lab.eta[perm[:n_cal]], lab.silent[perm[:n_cal]], lab.good[perm[:n_cal]], lab.converged[perm[:n_cal]])
        test = type(lab)(lab.eta[perm[n_cal:]], lab.silent[perm[n_cal:]], lab.good[perm[n_cal:]], lab.converged[perm[n_cal:]])
        for a in ALPHAS:
            tau = conformal_threshold(cal.eta[cal.silent], a)
            o = outcome(test, tau)
            row = np.array([o["miss"], o["silent"], o["false_alarm"], o["good"], o["flagged"], o["n"], float(o["tau_finite"]), 1.0])
            totals[f"cc{a}"] = totals.get(f"cc{a}", 0) + row
    return totals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    args = parser.parse_args()
    files = sorted(AUDIT_DIR.glob(f"{args.problem}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no audits for {args.problem}; run scripts/generate_audits.py first")
    audits = [load_audit(f) for f in files]
    models = [tuple(m) for m in audits[0].meta["settings"]["models"] if m[0] == "nmf"]
    rng, boot_rng = np.random.default_rng(SEEDS["split"]), np.random.default_rng(SEEDS["bootstrap"])
    out = {"problem": args.problem, "n_replicates": len(audits), "n_splits": N_SPLITS, "n_cal": N_CAL[args.problem],
           "eps_values": list(EPS_VALUES), "seeds": SEEDS, "environment": environment_info(), "cells": {}}
    for kind, r in models:
        for eps in EPS_VALUES:
            ps, aucs, per_rule = [], [], {}
            for a in audits:
                g = lambda f: a.arrays[f"pool|{kind}|{r}|{f}"]
                lab = labels(g("eta"), g("err"), g("converged"), eps)
                ps.append(float(lab.silent.mean()))
                value = auc(lab.eta[lab.silent], lab.eta[lab.good])
                if value is not None:
                    aucs.append(value)
                for rule, row in split_rows(lab, N_CAL[args.problem], rng).items():
                    per_rule.setdefault(rule, []).append(row)
            p = float(np.mean(ps))
            boot = np.mean(boot_rng.choice(ps, size=(N_BOOT, len(ps))), axis=1)
            cell = {"p": p, "p_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                    "auc": float(np.mean(aucs)) if aucs else None}
            for alpha in ALPHAS:
                cell[f"need@{alpha}"] = math.ceil(min_failures(alpha) / p) if p > 0 else None
            cell.update({rule: summarise(rows, boot_rng) for rule, rows in per_rule.items()})
            out["cells"][f"{kind}|{r}|{eps}"] = cell
        log.info("model %s %d done", kind, r)
    path = RESULTS / f"eps_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    f = lambda x: " -- " if x is None else f"{x:.3f}"
    print(f"\n=== eps sensitivity ({args.problem}, n_cal {N_CAL[args.problem]}): p | AUC | need@0.05 | cc0.1 miss | cc0.1 false alarm | defined")
    for kind, r in models:
        for eps in EPS_VALUES:
            c = out["cells"][f"{kind}|{r}|{eps}"]
            print(f"{kind}{r:<3} eps {eps:<5} p {f(c['p'])} auc {f(c['auc'])} need {c['need@0.05']} | "
                  f"{f(c['cc0.1']['miss'])} {f(c['cc0.1']['fpr'])} {c['cc0.1']['defined']:.2f}")


if __name__ == "__main__":
    main()
