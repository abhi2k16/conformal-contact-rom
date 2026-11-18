"""Experiment F: failure-enriched calibration with importance weights, against plain uniform calibration.

Silent failures can be rare, and the class-conditional rule needs ``ceil(1/alpha) - 1`` of them in the calibration set.
Idea: spend the calibration budget near the predicted failure region. Calibration parameters are drawn from the
mixture ``q = lam p + (1 - lam) p h / Z`` (``contactscreen.focus``), whose density ratio ``w = p / q <= 1/lam`` is known,
and the threshold is the weighted conformal one (``conformal.weighted_thresholds``) with the test query's own weight.
``lam`` is the weight of the defensive uniform component; a sweep over ``--lams`` shows the trade-off between
enrichment (small ``lam``) and bounded weights (large ``lam``).

Per replicate and rare-failure model: the in-range pool is split in half; the first half is the pilot (fits ``h``, and
supplies the uniform calibration queries of the baseline), the second half is the test set (uniform, as in
deployment). With a budget of ``n`` new full-order solves the calibrations are compared on the same test set:
  uniform           ``n`` queries from the pilot half (the baseline of ``run_coverage.py``);
  weighted@<lam>    ``n`` queries from ``q``, weighted rule (valid);
  naive@<lam>       the same ``n`` queries, unweighted rule (ignores the shift; kept to show what weighting buys).
Reported per cell: miss rate, false-alarm rate, flagged share, defined share (share of test queries with a finite
threshold; ``tau = -infinity`` flags every query), and per budget the mean number of calibration failures and the
effective sample size ``(sum w)^2 / sum w^2`` of the weighted failures.
    python scripts/run_enriched.py --problem membrane
Writes ``results/enriched_<problem>.json``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import build_models, load_audit                      # noqa: E402
from contactscreen.config import AUDIT_DIR, PROBLEMS, RESULTS, SEEDS, make_problem, make_settings    # noqa: E402
from contactscreen.conformal import EPS, conformal_threshold, effective_sample_size, weighted_thresholds   # noqa: E402
from contactscreen.environment import environment_info                       # noqa: E402
from contactscreen.focus import FailureModel, FocusProposal                  # noqa: E402
from contactscreen.fom import FullOrderSolver, energy_error                  # noqa: E402
from contactscreen.rom import admissibility_indicator                        # noqa: E402

log = logging.getLogger("run_enriched")

RARE = {"membrane": (("nmf", 36), ("nmf", 48)), "block": (("nmf", 16), ("nmf", 24)), "layered": (("nmf", 16), ("nmf", 24))}
BUDGETS = (40, 80, 160)
ALPHAS = (0.05, 0.10)
N_DRAWS = 3                       # independent calibration draws per (replicate, model, budget)


def evaluate(problem, solver, model, mus: np.ndarray) -> tuple:
    """Indicator values, silent-failure labels and converged flags of ``model`` at ``mus`` (one full-order solve each)."""
    eta, silent, conv = np.zeros(len(mus)), np.zeros(len(mus), bool), np.zeros(len(mus), bool)
    for i, mu in enumerate(mus):
        sol = solver.solve(mu)
        red = model.solve(mu, sol.K)
        eta[i] = admissibility_indicator(problem, mu, red)[0]
        conv[i] = red.converged
        silent[i] = red.converged and energy_error(sol.K, sol.u, red.u) > EPS
    return eta, silent, conv


def score(eta, silent, conv, tau) -> np.ndarray:
    """Counts ``[miss, silent, false alarm, good, flagged, n, defined]`` of a (per-query) threshold ``tau``."""
    tau = np.broadcast_to(tau, eta.shape)
    fl = ~conv | (eta >= tau)
    good = conv & ~silent
    return np.array([(silent & ~fl).sum(), silent.sum(), (good & fl).sum(), good.sum(), fl.sum(), len(eta), np.isfinite(tau).mean() * len(eta)])


def parse_lams(text: str) -> tuple:
    lams = tuple(float(x) for x in text.split(","))
    if any(not 0.0 < lam <= 1.0 for lam in lams):
        raise ValueError("--lams must be values in (0, 1]")
    return lams


def run_replicate(problem, solver, audit, args, lams, rng, acc) -> None:
    seed = audit.meta["settings"]["seed"]
    settings = make_settings(args.problem, seed)
    models = build_models(problem, solver, problem.sample(np.random.default_rng(seed), settings.n_train), settings)
    pool_mu = audit.arrays["pool|mu"]
    half = len(pool_mu) // 2
    for key in RARE[args.problem]:
        g = lambda f: audit.arrays[f"pool|{key[0]}|{key[1]}|{f}"]
        conv_all = g("converged") > 0.5
        silent_all = conv_all & (g("err") > EPS)
        eta_all = g("eta")
        pilot, test = slice(0, half), slice(half, None)
        failure_model = FailureModel.fit(pool_mu[pilot], silent_all[pilot], problem.lo, problem.hi)
        props = {lam: FocusProposal(failure_model, lam=lam) for lam in lams}
        t_eta, t_sil, t_conv = eta_all[test], silent_all[test], conv_all[test]
        for n in BUDGETS:
            for _ in range(N_DRAWS):
                idx = rng.permutation(half)[:n]
                uni_eta, uni_sil = eta_all[pilot][idx], silent_all[pilot][idx]
                acc.setdefault((key, n, "stats", "uniform"), []).append([uni_sil.sum(), 0.0])
                for lam, prop in props.items():
                    enr_mu = prop.sample(rng, n)
                    e_eta, e_sil, _ = evaluate(problem, solver, models[key], enr_mu)
                    w_enr = prop.weights(enr_mu)
                    w_test = prop.weights(pool_mu[test])
                    acc.setdefault((key, n, "stats", f"enriched@{lam}"), []).append([e_sil.sum(), effective_sample_size(w_enr[e_sil])])
                    for a in ALPHAS:
                        taus = {f"naive@{lam}": conformal_threshold(e_eta[e_sil], a),
                                f"weighted@{lam}": weighted_thresholds(e_eta[e_sil], w_enr[e_sil], w_test, a)}
                        if lam == lams[0]:
                            taus["uniform"] = conformal_threshold(uni_eta[uni_sil], a)
                        for rule, tau in taus.items():
                            acc.setdefault((key, n, a, rule), []).append(score(t_eta, t_sil, t_conv, tau))
        log.info("%s seed %d model %s done", args.problem, seed, key)


def summarise(acc: dict) -> tuple:
    cells, stats = {}, {}
    for k, rows in acc.items():
        if k[2] == "stats":
            arr = np.array(rows)
            stats[f"{k[0][0]}|{k[0][1]}|{k[1]}|{k[3]}"] = {"failures_in_calibration": float(arr[:, 0].mean()), "ess_of_weighted_failures": float(arr[:, 1].mean())}
            continue
        tot = np.sum(rows, axis=0)
        cells[f"{k[0][0]}|{k[0][1]}|{k[1]}|{k[2]}|{k[3]}"] = {
            "miss": float(tot[0] / tot[1]) if tot[1] else None, "fpr": float(tot[2] / tot[3]) if tot[3] else None,
            "flagged": float(tot[4] / tot[5]), "defined": float(tot[6] / tot[5])}
    return cells, stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    p.add_argument("--seeds", type=int, default=8, help="number of replicates used")
    p.add_argument("--lams", default="0.3,0.5,0.7", help="weights of the defensive uniform component, comma separated")
    args = p.parse_args()
    lams = parse_lams(args.lams)
    files = sorted(AUDIT_DIR.glob(f"{args.problem}_seed*.npz"))[:args.seeds]
    if not files:
        raise FileNotFoundError(f"no audits for {args.problem}; run scripts/generate_audits.py first")
    problem = make_problem(args.problem)
    solver = FullOrderSolver(problem)
    rng, acc = np.random.default_rng(SEEDS["enriched"]), {}
    for path in files:
        run_replicate(problem, solver, load_audit(path), args, lams, rng, acc)
    cells, stats = summarise(acc)
    out = {"problem": args.problem, "n_replicates": len(files), "budgets": list(BUDGETS), "lams": list(lams), "n_draws": N_DRAWS,
           "seeds": SEEDS, "environment": environment_info(), "cells": cells, "calibration_sets": stats}
    path = RESULTS / f"enriched_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    print_table(out, RARE[args.problem])


def print_table(out: dict, models: tuple) -> None:
    fmt = lambda x: "  --" if x is None else f"{x:.3f}"
    for key in models:
        print(f"\n=== {out['problem']} {key[0]}{key[1]}: miss rate | false-alarm rate | defined share")
        for n in out["budgets"]:
            u = out["calibration_sets"][f"{key[0]}|{key[1]}|{n}|uniform"]
            print(f"  budget {n}: failures in calibration set: uniform {u['failures_in_calibration']:.1f}")
            for lam in out["lams"]:
                e = out["calibration_sets"][f"{key[0]}|{key[1]}|{n}|enriched@{lam}"]
                print(f"    lam {lam}: enriched {e['failures_in_calibration']:.1f} failures, ESS {e['ess_of_weighted_failures']:.1f}")
            for a in ALPHAS:
                for rule in ["uniform"] + [f"{r}@{lam}" for lam in out["lams"] for r in ("naive", "weighted")]:
                    c = out["cells"][f"{key[0]}|{key[1]}|{n}|{a}|{rule}"]
                    print(f"    alpha {a:<5}{rule:<14} miss {fmt(c['miss'])}  fpr {fmt(c['fpr'])}  defined {c['defined']:.2f}")


if __name__ == "__main__":
    main()
