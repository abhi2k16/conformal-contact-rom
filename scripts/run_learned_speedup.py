"""Experiment J: net speedup of the screened pipeline with the learned score, against the standard indicator (``layered``).

Same pipeline cost model as experiment E (``run_fallback.py``): every query pays the reduced solve and the score; a
flagged query also pays a full-order fallback (cold, or warm-started with the seed threshold chosen in experiment D,
read from ``results/fallback_<problem>.json``); speedup = mean cold full-order time / mean pipeline time. The learned
score additionally pays its feature computation (timed). Differences from E: ``--n-time`` fresh timing queries are all
used for evaluation (no tuning half), and thresholds (and the learned score) are redrawn ``--draws`` times, so the
calibration-draw variability is reported instead of a single draw.
Every variant spends the same full-order budget ``T``: ``eta@T`` calibrates on ``T`` pool queries; ``learned@T:F`` fits on
``F`` fresh queries (``generate_features.py``, tag ``fit``) and calibrates on ``T - F`` pool queries. One thread; run alone:
    python scripts/run_learned_speedup.py --problem layered
Writes ``results/learned_speedup_<problem>.json``. Needs the audit, features and ``fallback_<problem>.json`` of replicate 0.
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

from contactscreen.audit import build_models                                      # noqa: E402
from contactscreen.config import PROBLEMS, RESULTS, SEEDS, features_path, make_problem, make_settings   # noqa: E402
from contactscreen.conformal import (QueryLabels, conformal_threshold, flagged, labels, risk_threshold)   # noqa: E402
from contactscreen.features import FeatureScorer, query_features                  # noqa: E402
from contactscreen.fom import FullOrderSolver, energy_error, time_call           # noqa: E402
from contactscreen.rom import admissibility_indicator                            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fallback import pipeline_row, seed_mask                                  # noqa: E402

log = logging.getLogger("run_learned_speedup")

BUDGET = 160
FIT_SIZES = (40, 80)
RULES = {"cc0.05": lambda lab: conformal_threshold(lab.eta[lab.silent], 0.05),
         "cc0.1": lambda lab: conformal_threshold(lab.eta[lab.silent], 0.10),
         "crc0.01": lambda lab: risk_threshold(lab.eta, lab.silent, 0.01),
         "flag_all": lambda lab: -np.inf}


def measure(problem, solver, models, rels: dict, mus, repeats: int) -> dict:
    """Per query: cold full-order time; per model: reduced solve, score and feature times, error, warm solve time."""
    nq = len(mus)
    t_cold = np.zeros(nq)
    data = {k: {"eta": np.zeros(nq), "err": np.zeros(nq), "conv": np.zeros(nq), "t_rom": np.zeros(nq), "t_feat": np.zeros(nq),
                "t_warm": np.zeros(nq), "X": []} for k in models}
    for i, mu in enumerate(mus):
        t_cold[i], cold = time_call(lambda: solver.solve(mu), repeats)
        for key, model in models.items():
            d = data[key]
            t_rom, red = time_call(lambda: model.solve(mu, cold.K), repeats)
            eta, t_eta = admissibility_indicator(problem, mu, red)
            t_feat, x = time_call(lambda: query_features(problem, mu, cold.K, red), repeats)
            d["eta"][i], d["conv"][i], d["t_rom"][i], d["t_feat"][i] = eta, float(red.converged), t_rom + t_eta, t_feat
            d["err"][i] = energy_error(cold.K, cold.u, red.u)
            d["X"].append(x)
            seed = seed_mask(red.lam, rels[key])
            d["t_warm"][i] = time_call(lambda: solver.solve(mu, active0=seed), repeats)[0]
        if (i + 1) % 10 == 0:
            log.info("%d/%d queries timed", i + 1, nq)
    for d in data.values():
        d["X"] = np.array(d["X"])
    return {"t_cold": t_cold, "models": data}


def draw_rows(feat, key, d, t_cold, rng) -> dict:
    """One calibration draw: ``{variant: {rule: row}}`` for every variant of one model."""
    g = lambda tag, f: feat[f"{tag}|{key[0]}|{key[1]}|{f}"]
    pool_lab = labels(g("pool", "eta"), g("pool", "err"), g("pool", "converged"))
    fit_lab = labels(g("fit", "eta"), g("fit", "err"), g("fit", "converged"))
    test_eta = labels(d["eta"], d["err"], d["conv"])
    perm = rng.permutation(len(pool_lab.eta))
    out, everything = {}, slice(None)

    def evaluate(variant, cal_scores, test_scores, t_extra):
        cal = QueryLabels(cal_scores, pool_lab.silent[cal_idx], pool_lab.good[cal_idx], pool_lab.converged[cal_idx])
        test = QueryLabels(test_scores, test_eta.silent, test_eta.good, test_eta.converged)
        dd = {"t_rom": d["t_rom"] + t_extra}
        out[variant] = {rule: {"tau_defined": bool(np.isfinite(tau)),
                               **pipeline_row(flagged(test, tau), test, dd, t_cold, d["t_warm"], everything)}
                        for rule, fn in RULES.items() for tau in [fn(cal)]}

    cal_idx = perm[:BUDGET]
    evaluate(f"eta@{BUDGET}", g("pool", "eta")[cal_idx], d["eta"], 0.0)
    for n_fit in FIT_SIZES:
        rows = rng.permutation(len(fit_lab.eta))[:n_fit]
        rows = rows[fit_lab.converged[rows]]
        scorer = FeatureScorer.fit(g("fit", "X")[rows], fit_lab.silent[rows])
        if scorer is None:
            continue
        cal_idx = perm[:BUDGET - n_fit]
        evaluate(f"learned@{BUDGET}:{n_fit}", scorer.score(g("pool", "X")[cal_idx]), scorer.score(d["X"]), d["t_feat"])
    return out


def aggregate(rows: list) -> dict:
    """Mean and spread over draws of one (variant, rule) cell."""
    arr = lambda k: np.array([r[k] for r in rows], float)
    silent, missed = arr("n_silent").sum(), arr("missed").sum()
    return {"draws": len(rows), "defined": float(np.mean([r["tau_defined"] for r in rows])),
            "miss_rate": float(missed / silent) if silent else None, "fallback": float(arr("fallback").mean()),
            "speedup_cold": float(arr("speedup_cold").mean()), "speedup_cold_sd": float(arr("speedup_cold").std()),
            "speedup_warm": float(arr("speedup_warm").mean()), "speedup_warm_sd": float(arr("speedup_warm").std())}


def summarise(m: dict, feat, draws: int, rng) -> dict:
    out = {}
    for key, d in m["models"].items():
        per: dict = {}
        for _ in range(draws):
            for variant, rules in draw_rows(feat, key, d, m["t_cold"], rng).items():
                for rule, row in rules.items():
                    per.setdefault(f"{variant}|{rule}", []).append(row)
        test = labels(d["eta"], d["err"], d["conv"])
        oracle = pipeline_row(~test.converged | test.silent, test, {"t_rom": d["t_rom"]}, m["t_cold"], d["t_warm"], slice(None))
        out[f"{key[0]}|{key[1]}"] = {"silent_rate": float(test.silent.mean()), "n_silent": int(test.silent.sum()),
                                     "t_cold_ms_mean": 1e3 * float(m["t_cold"].mean()), "t_warm_ms_mean": 1e3 * float(d["t_warm"].mean()),
                                     "t_rom_ms_mean": 1e3 * float(d["t_rom"].mean()), "t_feat_ms_mean": 1e3 * float(d["t_feat"].mean()),
                                     "oracle": oracle, "cells": {k: aggregate(v) for k, v in per.items()}}
    return out


def print_tables(r: dict) -> None:
    print(f"\n=== J ({r['problem']}): {r['n_time']} timing queries, {r['draws']} calibration draws, budget T = {BUDGET}; speedup cold | warm (sd), miss rate, fallback share, rule defined")
    for name, v in r["models"].items():
        print(f"-- {name}: silent rate {v['silent_rate']:.2f} ({v['n_silent']}); cold {v['t_cold_ms_mean']:.0f} ms, warm {v['t_warm_ms_mean']:.0f} ms, "
              f"ROM+eta {v['t_rom_ms_mean']:.1f} ms, features {v['t_feat_ms_mean']:.1f} ms; oracle {v['oracle']['speedup_cold']:.1f} | {v['oracle']['speedup_warm']:.1f}")
        for cell, c in v["cells"].items():
            miss = "--" if c["miss_rate"] is None else f"{c['miss_rate']:.3f}"
            print(f"   {cell:<24}{c['speedup_cold']:>6.2f} ({c['speedup_cold_sd']:.2f}) |{c['speedup_warm']:>6.2f} ({c['speedup_warm_sd']:.2f})   miss {miss}   fallback {c['fallback']:.2f}   defined {c['defined']:.2f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), default="layered")
    p.add_argument("--n-time", type=int, default=120)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--draws", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="replicate whose models, features and seed thresholds are used")
    args = p.parse_args()
    fallback_json, feature_file = RESULTS / f"fallback_{args.problem}.json", features_path(args.problem, args.seed)
    for path in (fallback_json, feature_file):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run run_fallback.py / generate_features.py first")
    chosen = json.loads(fallback_json.read_text(encoding="utf-8"))["D"]["models"]
    problem, settings = make_problem(args.problem), make_settings(args.problem, args.seed)
    solver = FullOrderSolver(problem)
    models = {k: m for k, m in build_models(problem, solver, problem.sample(np.random.default_rng(settings.seed), settings.n_train),
                                            settings).items() if k[0] == "nmf"}
    rels = {k: chosen[f"{k[0]}|{k[1]}"]["chosen_rel"] for k in models}
    mus = problem.sample(np.random.default_rng(SEEDS["timing_queries"] + args.seed), args.n_time)
    log.info("%s: N=%d, %d timing queries, seed thresholds %s", problem.name, problem.n_dof, args.n_time, rels)
    m = measure(problem, solver, models, rels, mus, args.repeats)
    feat = np.load(feature_file)
    result = {"problem": args.problem, "n_time": args.n_time, "repeats": args.repeats, "draws": args.draws, "budget": BUDGET,
              "fit_sizes": list(FIT_SIZES), "seed_rels": {f"{k[0]}|{k[1]}": v for k, v in rels.items()}, "seeds": SEEDS,
              "environment": environment_info(), "models": summarise(m, feat, args.draws, np.random.default_rng(SEEDS["split"]))}
    out_path = RESULTS / f"learned_speedup_{args.problem}.json"
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    log.info("wrote %s", out_path)
    print_tables(result)


if __name__ == "__main__":
    main()
