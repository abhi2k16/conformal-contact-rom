"""Experiments D and E: cost of the full-order fallback with and without a warm start, and the net speedup.

Uses ``fea_engine.solve_contact_static_sparse`` for the full-order solve (cold start, or warm start ``active0``) and
``rom_engine.ContactROM`` for the reduced solve. BLAS and OpenMP are pinned to one thread before NumPy is imported.
The warm-start seed is the set of constraints whose lifted reduced multiplier exceeds ``rel * max(lifted multiplier)``
for ``rel`` in ``SEED_RELS`` (``rel = 0``: every positive entry).

The ``--n-time`` fresh in-range queries are split in two halves. The first half (tuning) picks, per model, the seed
threshold with the smallest median warm-solve time; everything reported uses the second half (evaluation).
D. Per query: iterations and best-of-``--repeats`` wall time of the cold and the warm full-order solve for every seed
   threshold; overlap (Jaccard) of seed and converged active set; difference between the two solutions; stiffness
   assembly time ``K(mu)`` (non-zero only when ``K`` varies); time of one sparse LU of the free block (the suite solver
   is one call, so the factorisation inside it cannot be timed directly: time per iteration and this factorisation time
   are the proxy); number of solves that needed a fallback route.
E. Net speedup of the screened pipeline per rule: every query pays the reduced solve and the indicator; a flagged query
   additionally pays a fallback (cold, or warm with the chosen seed). Speedup = mean cold full-order time / mean
   pipeline time. Reference lines: the *oracle* screen flags exactly the silent failures (upper bound on the speedup
   at zero miss rate); a *random* screen with the same flag share has the same speedup in expectation but misses a
   fraction ``1 - share`` of the silent failures, which is what shows the value of the indicator. Thresholds are
   calibrated on 160 queries of the saved audit pool of replicate 0 (same models); the timed queries are fresh, so
   they are exchangeable with the calibration queries.

Run one process at a time (timings), from the project folder:
    python scripts/run_fallback.py --problem block
Writes ``results/fallback_<problem>.json``. Needs ``scripts/generate_audits.py`` output for replicate 0.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.environment import environment_info, single_thread            # noqa: E402

single_thread()                                                                   # before NumPy is imported

import argparse                                                                   # noqa: E402
import json                                                                       # noqa: E402
import logging                                                                    # noqa: E402

import numpy as np                                                                # noqa: E402
import scipy.sparse as sp                                                         # noqa: E402
import scipy.sparse.linalg as spla                                                # noqa: E402

from contactscreen.audit import build_models, load_audit                         # noqa: E402
from contactscreen.config import PROBLEMS, RESULTS, SEEDS, audit_path, make_problem, make_settings   # noqa: E402
from contactscreen.conformal import (conformal_threshold, flagged, heuristic_threshold, labels,   # noqa: E402
                                     risk_threshold)
from contactscreen.fom import FullOrderSolver, energy_error, time_call           # noqa: E402
from contactscreen.rom import admissibility_indicator                            # noqa: E402

log = logging.getLogger("run_fallback")

N_CAL = 160
SEED_RELS = (0.0, 0.01, 0.05, 0.2)
RULES = {"cc0.05": lambda lab: (conformal_threshold(lab.eta[lab.silent], 0.05), False),
         "cc0.1": lambda lab: (conformal_threshold(lab.eta[lab.silent], 0.10), False),
         "crc0.01": lambda lab: (risk_threshold(lab.eta, lab.silent, 0.01), False),
         "heur": lambda lab: (heuristic_threshold(lab.eta[lab.silent]), True),
         "flag_all": lambda lab: (-np.inf, False)}


def seed_mask(lam: np.ndarray, rel: float) -> np.ndarray:
    return lam > rel * max(float(lam.max()), 0.0) if rel > 0 else lam > 0


def factorisation_time(problem, K: sp.spmatrix, active_mask: np.ndarray, repeats: int) -> float:
    """Best-of time of one sparse LU of the free block (unknowns not fixed by an active constraint)."""
    free = np.setdiff1d(np.arange(problem.n_dof), problem.B.indices[active_mask])
    block = sp.csc_matrix(K[free][:, free])
    return time_call(lambda: spla.splu(block), repeats)[0]


def measure(problem, solver, models, mus, repeats: int) -> dict:
    """Per query: cold solve and its parts; per model the reduced solve, indicator, error, warm solves per seed threshold."""
    nq, nr = len(mus), len(SEED_RELS)
    fom = {k: np.zeros(nq) for k in ("it_cold", "t_cold", "t_assembly", "t_factor", "fallbacks_cold")}
    data = {k: {"eta": np.zeros(nq), "err": np.zeros(nq), "conv": np.zeros(nq), "t_rom": np.zeros(nq),
                "it_warm": np.zeros((nr, nq)), "t_warm": np.zeros((nr, nq)), "jaccard": np.zeros((nr, nq)),
                "du": np.zeros((nr, nq)), "fallbacks_warm": np.zeros((nr, nq))} for k in models}
    for i, mu in enumerate(mus):
        fom["t_assembly"][i] = time_call(lambda: problem.stiffness(mu), repeats)[0]
        t_cold, cold = time_call(lambda: solver.solve(mu), repeats)
        fom["it_cold"][i], fom["t_cold"][i], fom["fallbacks_cold"][i] = cold.iterations, t_cold, cold.fallbacks
        fom["t_factor"][i] = factorisation_time(problem, cold.K, cold.active_mask, repeats)
        for key, model in models.items():
            t_rom, red = time_call(lambda: model.solve(mu, cold.K), repeats)
            eta, t_eta = admissibility_indicator(problem, mu, red)
            d = data[key]
            d["eta"][i], d["conv"][i], d["t_rom"][i] = eta, float(red.converged), t_rom + t_eta
            d["err"][i] = energy_error(cold.K, cold.u, red.u)
            for j, rel in enumerate(SEED_RELS):
                mask = seed_mask(red.lam, rel)
                t_warm, warm = time_call(lambda: solver.solve(mu, active0=mask), repeats)
                d["it_warm"][j, i], d["t_warm"][j, i], d["fallbacks_warm"][j, i] = warm.iterations, t_warm, warm.fallbacks
                d["jaccard"][j, i] = (mask & cold.active_mask).sum() / max(1, (mask | cold.active_mask).sum())
                d["du"][j, i] = np.max(np.abs(warm.u - cold.u)) / max(1e-300, np.max(np.abs(cold.u)))
        if (i + 1) % 10 == 0:
            log.info("%d/%d queries timed", i + 1, nq)
    return {"fom": fom, "models": data}


def chosen_rel(d: dict, tune: slice) -> int:
    """Index of the seed threshold with the smallest median warm time on the tuning half."""
    return int(np.argmin(np.median(d["t_warm"][:, tune], axis=1)))


def summarise_d(m: dict, tune: slice, evl: slice) -> dict:
    fom = m["fom"]
    med = lambda a: float(np.median(a[evl]))
    out = {"seed_rels": list(SEED_RELS),
           "fom": {"it_cold_median": med(fom["it_cold"]), "t_cold_ms_median": 1e3 * med(fom["t_cold"]),
                   "t_per_iteration_ms_median": 1e3 * med(fom["t_cold"] / np.maximum(fom["it_cold"], 1)),
                   "t_assembly_ms_median": 1e3 * med(fom["t_assembly"]), "t_factor_free_block_ms_median": 1e3 * med(fom["t_factor"]),
                   "solves_with_fallback_route": int(fom["fallbacks_cold"][evl].sum())}, "models": {}}
    for key, d in m["models"].items():
        rows = []
        for j, rel in enumerate(SEED_RELS):
            rows.append({"rel": rel, "it_warm_median": float(np.median(d["it_warm"][j, evl])),
                         "share_fewer": float(np.mean(d["it_warm"][j, evl] < fom["it_cold"][evl])),
                         "share_more": float(np.mean(d["it_warm"][j, evl] > fom["it_cold"][evl])),
                         "t_warm_ms_median": 1e3 * float(np.median(d["t_warm"][j, evl])),
                         "jaccard_median": float(np.median(d["jaccard"][j, evl])), "max_du": float(d["du"][j, evl].max()),
                         "solves_with_fallback_route": int(d["fallbacks_warm"][j, evl].sum())})
        out["models"][f"{key[0]}|{key[1]}"] = {"t_rom_ms_median": 1e3 * float(np.median(d["t_rom"][evl])),
                                               "chosen_rel": SEED_RELS[chosen_rel(d, tune)], "by_rel": rows}
    return out


def pipeline_row(fl: np.ndarray, test, d: dict, t_cold: np.ndarray, t_warm: np.ndarray, evl: slice) -> dict:
    """Fallback share, misses and speedups of one flagging decision ``fl`` on the evaluation queries."""
    cost = {"cold": d["t_rom"][evl] + fl * t_cold, "warm": d["t_rom"][evl] + fl * t_warm}
    share = float(fl.mean())
    return {"fallback": share, "missed": int((test.silent & ~fl).sum()), "n_silent": int(test.silent.sum()),
            "random_miss_rate_at_same_share": 1.0 - share,
            "speedup_cold": float(t_cold.mean() / cost["cold"].mean()), "speedup_warm": float(t_cold.mean() / cost["warm"].mean())}


def summarise_e(m: dict, tune: slice, evl: slice, cal_audit, rng) -> dict:
    """Net speedup per model and rule (plus the oracle) on the evaluation half; thresholds from ``N_CAL`` audit-pool queries."""
    out, t_cold = {}, m["fom"]["t_cold"][evl]
    for key, d in m["models"].items():
        g = lambda f: cal_audit.arrays[f"pool|{key[0]}|{key[1]}|{f}"]
        pool = labels(g("eta"), g("err"), g("converged"))
        idx = rng.permutation(len(pool.eta))[:N_CAL]
        cal = type(pool)(pool.eta[idx], pool.silent[idx], pool.good[idx], pool.converged[idx])
        test = labels(d["eta"][evl], d["err"][evl], d["conv"][evl])
        t_warm = d["t_warm"][chosen_rel(d, tune)][evl]
        for rule, fn in RULES.items():
            tau, strict = fn(cal)
            row = pipeline_row(flagged(test, tau, strict), test, d, t_cold, t_warm, evl)
            out[f"{key[0]}|{key[1]}|{rule}"] = {"tau_defined": bool(np.isfinite(tau)), **row}
        out[f"{key[0]}|{key[1]}|oracle"] = {"tau_defined": True, **pipeline_row(~test.converged | test.silent, test, d, t_cold, t_warm, evl)}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    p.add_argument("--n-time", type=int, default=60)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="replicate whose models and calibration pool are used")
    args = p.parse_args()
    if args.n_time < 4 or args.repeats < 1:
        raise ValueError("--n-time must be at least 4 (two halves) and --repeats positive")
    path = audit_path(args.problem, args.seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/generate_audits.py first")
    problem, settings = make_problem(args.problem), make_settings(args.problem, args.seed)
    solver = FullOrderSolver(problem)
    models = build_models(problem, solver, problem.sample(np.random.default_rng(settings.seed), settings.n_train), settings)
    mus = problem.sample(np.random.default_rng(SEEDS["timing_queries"] + args.seed), args.n_time)
    log.info("%s: N=%d, n_c=%d, timing %d queries", problem.name, problem.n_dof, problem.n_contact, args.n_time)
    m = measure(problem, solver, models, mus, args.repeats)
    half = args.n_time // 2
    tune, evl = slice(0, half), slice(half, None)
    result = {"problem": args.problem, "n_time": args.n_time, "repeats": args.repeats, "n_eval": args.n_time - half,
              "seeds": SEEDS, "replicate": args.seed, "environment": environment_info(),
              "D": summarise_d(m, tune, evl),
              "E": summarise_e(m, tune, evl, load_audit(path), np.random.default_rng(SEEDS["split"]))}
    out_path = RESULTS / f"fallback_{args.problem}.json"
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    log.info("wrote %s", out_path)
    print_tables(result)


def print_tables(r: dict) -> None:
    d = r["D"]["fom"]
    print(f"\n=== D ({r['problem']}, {r['n_eval']} evaluation queries): cold full-order solve median {d['it_cold_median']:.0f} iterations, "
          f"{d['t_cold_ms_median']:.1f} ms ({d['t_per_iteration_ms_median']:.1f} ms/iteration); K(mu) assembly {d['t_assembly_ms_median']:.2f} ms; "
          f"one LU of the free block {d['t_factor_free_block_ms_median']:.1f} ms; solves needing a fallback route: {d['solves_with_fallback_route']}")
    print(f"{'model':<8}{'rel':>6}{'it warm':>9}{'fewer':>7}{'more':>6}{'ms warm':>9}{'Jaccard':>9}{'max du':>9}   (chosen on tuning half: *)")
    for name, v in r["D"]["models"].items():
        for row in v["by_rel"]:
            star = "*" if row["rel"] == v["chosen_rel"] else " "
            print(f"{name:<8}{row['rel']:>6}{row['it_warm_median']:>9.1f}{row['share_fewer']:>7.2f}{row['share_more']:>6.2f}"
                  f"{row['t_warm_ms_median']:>9.1f}{row['jaccard_median']:>9.2f}{row['max_du']:>9.1e} {star}")
    print("=== E: speedup (cold | warm fallback), fallback share, missed / silent failures, miss rate of a random screen at the same share")
    for name, v in r["E"].items():
        print(f"{name:<18} {v['speedup_cold']:>7.2f} {v['speedup_warm']:>7.2f}   fallback {v['fallback']:.2f}   missed {v['missed']}/{v['n_silent']}"
              f"   random would miss {v['random_miss_rate_at_same_share']:.2f}")


if __name__ == "__main__":
    main()
