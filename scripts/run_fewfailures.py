"""Experiments L and M: calibration of the class-conditional rule when silent failures are rare (research_plan.md §8).

L, lower-threshold calibration. Failures at a lower tolerance eps' < eps are more frequent. Since every failure at eps
is one at eps', calibrating cc at level alpha' on the eps'-failures gives P(miss | e > eps) <= alpha' / r with
r = P(e > eps | e > eps'). The rule targets alpha at eps with alpha' = alpha * r, where r is either
  * ``exact_ratio``: exact pool-wide ratio p/p' of the whole replicate pool (uses test information; retained as an internal variant name), or
  * ``cp``: a one-sided Clopper-Pearson lower bound on r from the calibration set at level ``BETA``, so that the bound
    holds with probability at least 1 - BETA over the calibration draw (a PAC-type statement, not a marginal one).
The expected calibration size is (1/(alpha r) - 1) r / p' ... = 1/(alpha p) - r/p, never below the standard
1/(alpha p) - 1/p, so L cannot lower the cost; the experiment measures how it behaves at a fixed calibration size.

M, sequential calibration. Queries of a random order are added to the calibration set until it holds
``m_j = ceil(j / alpha) - 1`` silent failures, so that the threshold is the j-th smallest failure score (j in ``M_RANKS``;
j = 1 gives m = n_min); the cc threshold is then always defined, and a larger j trades calibration solves for fewer false alarms. The failures of an i.i.d. stream remain
exchangeable with a new failure, so the guarantee is unchanged; the price is a random calibration size. If the pool
runs out first, the rule is undefined (every query flagged) and the split is counted as exhausted.

Both on the saved audits (in-range pool, indicator eta), replicates as independent units, as run_coverage.py.
    python scripts/run_fewfailures.py --problem block
Writes ``results/fewfailures_<problem>.json``.
"""

import argparse
import json
import math
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import load_audit                                           # noqa: E402
from contactscreen.config import AUDIT_DIR, PROBLEMS, RESULTS, SEEDS               # noqa: E402
from contactscreen.conformal import EPS, conformal_threshold, min_failures          # noqa: E402
from contactscreen.environment import environment_info                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import COLS, N_SPLITS, summarise                                 # noqa: E402

log = logging.getLogger("run_fewfailures")

ALPHAS = (0.05, 0.10)
EPS_LOW = (0.01, 0.02, 0.03, 0.04)
BETA = 0.10                                             # confidence of the Clopper-Pearson bound in L
N_CAL = {"membrane": 160, "block": 160, "layered": 80}  # as experiment A
M_RANKS = (1, 2, 4)                                     # sequential calibration stops when the threshold rank reaches j
N_ORDERS_M = 1000                                      # random query orders per replicate for M (100 gave Monte Carlo errors near 0.01)


def counts(err_test, conv_test, flags, defined) -> np.ndarray:
    """Row in run_coverage.COLS order for the target tolerance EPS; reduced-solver failures always flagged."""
    silent, good = conv_test & (err_test > EPS), conv_test & (err_test <= EPS)
    fl = flags | ~conv_test
    return np.array([(silent & ~fl).sum(), silent.sum(), (good & fl).sum(), good.sum(), fl.sum(), len(fl),
                     float(defined), 1.0], float)


def cp_lower(successes: int, trials: int, beta: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound for a binomial proportion."""
    return 0.0 if successes == 0 else float(beta_dist.ppf(beta, successes, trials - successes + 1))


def lower_threshold_rows(eta, err, conv, cal, test, r_pool: dict) -> dict:
    """Experiment L for one split: {variant|alpha|eps': counts}."""
    out = {}
    for alpha in ALPHAS:
        tau = conformal_threshold(eta[cal][conv[cal] & (err[cal] > EPS)], alpha)
        out[f"standard|{alpha}|{EPS}"] = counts(err[test], conv[test], eta[test] >= tau, np.isfinite(tau))
        for eps_low in EPS_LOW:
            fail_low = conv[cal] & (err[cal] > eps_low)
            n_low, n_high = int(fail_low.sum()), int((conv[cal] & (err[cal] > EPS)).sum())
            for variant, r in (("exact_ratio", r_pool[eps_low]), ("cp", cp_lower(n_high, n_low, BETA) if n_low else 0.0)):
                tau = conformal_threshold(eta[cal][fail_low], alpha * r) if r > 0 else -np.inf
                out[f"{variant}|{alpha}|{eps_low}"] = counts(err[test], conv[test], eta[test] >= tau, np.isfinite(tau))
    return out


def sequential_rows(eta, err, conv, order) -> tuple:
    """Experiment M for one random order: ({stop|alpha: counts}, {stop|alpha: calibration size or None})."""
    rows, sizes = {}, {}
    failing = conv[order] & (err[order] > EPS)
    cum = np.cumsum(failing)
    for alpha in ALPHAS:
        for j in M_RANKS:
            m = math.ceil(j / alpha - 1e-9) - 1
            key = f"j{j}|{alpha}"
            hit = np.flatnonzero(cum >= m)
            if hit.size == 0 or hit[0] + 1 >= len(order):           # pool exhausted: no threshold, no test set left
                test = order
                rows[key] = counts(err[test], conv[test], np.ones(len(test), bool), False)
                sizes[key] = None
                continue
            n_cal = int(hit[0]) + 1
            cal, test = order[:n_cal], order[n_cal:]
            tau = conformal_threshold(eta[cal][conv[cal] & (err[cal] > EPS)], alpha)
            rows[key] = counts(err[test], conv[test], eta[test] >= tau, np.isfinite(tau))
            sizes[key] = n_cal
    return rows, sizes


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
    n_cal = N_CAL[args.problem]
    cells_l, cells_m, sizes_m, ratios = {}, {}, {}, {}
    for audit in audits:
        for kind, r in models:
            g = lambda f: audit.arrays[f"pool|{kind}|{r}|{f}"]
            eta, err, conv = g("eta"), g("err"), g("converged") > 0.5
            p = float((conv & (err > EPS)).mean())
            r_pool = {e: (p / float((conv & (err > e)).mean()) if (conv & (err > e)).any() else 0.0) for e in EPS_LOW}
            for e, v in r_pool.items():
                ratios.setdefault(f"{kind}|{r}|{e}", []).append(v)
            tot_l, tot_m, sz = {}, {}, {}
            for _ in range(N_SPLITS):
                order = rng.permutation(len(eta))
                for k, row in lower_threshold_rows(eta, err, conv, order[:n_cal], order[n_cal:], r_pool).items():
                    tot_l[k] = tot_l.get(k, 0) + row
            for _ in range(N_ORDERS_M):              # more orders: the sequential rule's split-to-split spread is large
                rows, sizes = sequential_rows(eta, err, conv, rng.permutation(len(eta)))
                for k, row in rows.items():
                    tot_m[k] = tot_m.get(k, 0) + row
                    sz.setdefault(k, []).append(sizes[k])
            for k, row in tot_l.items():
                cells_l.setdefault(f"{kind}|{r}|{k}", []).append(row)
            for k, row in tot_m.items():
                cells_m.setdefault(f"{kind}|{r}|{k}", []).append(row)
            for k, v in sz.items():
                sizes_m.setdefault(f"{kind}|{r}|{k}", []).extend(v)
        log.info("replicate done")
    size_summary = {}
    for k, v in sizes_m.items():
        done = [x for x in v if x is not None]
        size_summary[k] = {"exhausted": 1.0 - len(done) / len(v), "mean": float(np.mean(done)) if done else None,
                           "p90": float(np.percentile(done, 90)) if done else None}
    out = {"problem": args.problem, "n_replicates": len(audits), "n_splits": N_SPLITS, "n_cal_L": n_cal,
           "eps": EPS, "eps_low": list(EPS_LOW), "beta": BETA, "m_ranks": list(M_RANKS), "n_orders_M": N_ORDERS_M,
           "n_pool": int(len(audits[0].arrays["pool|mu"])), "seeds": SEEDS, "environment": environment_info(),
           "ratio_r": {k: float(np.mean(v)) for k, v in ratios.items()},
           "L": {k: summarise(v, boot_rng) for k, v in cells_l.items()},
           "M": {k: summarise(v, boot_rng) for k, v in cells_m.items()}, "M_sizes": size_summary}
    path = RESULTS / f"fewfailures_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    print_tables(out, models)


def print_tables(out: dict, models: list) -> None:
    f = lambda x: "  -- " if x is None else f"{x:.3f}"
    print(f"\n=== L ({out['problem']}, n_cal {out['n_cal_L']}): miss | false alarm | defined at eps = {out['eps']}")
    for kind, r in models:
        for alpha in ALPHAS:
            c = out["L"][f"{kind}|{r}|standard|{alpha}|{out['eps']}"]
            line = f"{kind}{r:<3} a={alpha:<5} standard {f(c['miss'])} {f(c['fpr'])} {c['defined']:.2f}"
            for e in EPS_LOW:
                for v in ("exact_ratio", "cp"):
                    c = out["L"][f"{kind}|{r}|{v}|{alpha}|{e}"]
                    line += f" | {v}@{e} {f(c['miss'])} {f(c['fpr'])} {c['defined']:.2f}"
            print(line)
    print(f"=== M (pool {out['n_pool']}): miss | false alarm | defined; calibration size mean / p90; share exhausted")
    for kind, r in models:
        for alpha in ALPHAS:
            for j in out["m_ranks"]:
                k = f"{kind}|{r}|j{j}|{alpha}"
                c, s = out["M"][k], out["M_sizes"][k]
                print(f"{kind}{r:<3} a={alpha:<5} j={j}  {f(c['miss'])} [{f(c['miss_ci'][0] if c['miss_ci'] else None)},{f(c['miss_ci'][1] if c['miss_ci'] else None)}] {f(c['fpr'])} {c['defined']:.2f}  "
                      f"size {s['mean'] if s['mean'] is None else round(s['mean'])} / {s['p90'] if s['p90'] is None else round(s['p90'])}  exhausted {s['exhausted']:.2f}")


if __name__ == "__main__":
    main()
