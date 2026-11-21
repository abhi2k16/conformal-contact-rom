"""Experiment H: does a much cheaper full-order solve remove the room for screening? (constant-stiffness problems)

Compares, on the same fresh queries as ``run_fallback.py`` (seed ``SEEDS['timing_queries']``):
  * the suite solver ``solve_contact_static_sparse`` (cold start),
  * the condensed Delassus solver of ``contactscreen/condensed.py`` (cold start, and warm-started from the lifted
    reduced multiplier with the seed threshold chosen on the tuning half, as in ``run_fallback.py``).
Reports the set-up cost and its break-even number of queries, per-query times and their split, agreement of the
answers, and the net speedup of the screened pipeline *against the condensed solve as the baseline* (the honest
question: is screening still worth it when the full-order solve is already fast?). BLAS and OpenMP are pinned to one
thread before NumPy is imported. Run alone (timings), from the project folder:
    python scripts/run_condensed.py --problem block
Writes ``results/condensed_<problem>.json``. Needs ``scripts/generate_audits.py`` output for the replicate.
Only for problems with constant stiffness (``membrane``, ``block``).
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

from contactscreen.audit import build_models, load_audit                         # noqa: E402
from contactscreen.condensed import CondensedSolver                              # noqa: E402
from contactscreen.config import PROBLEMS, RESULTS, SEEDS, audit_path, make_problem, make_settings   # noqa: E402
from contactscreen.conformal import flagged, labels                              # noqa: E402
from contactscreen.fom import FullOrderSolver, energy_error, time_call           # noqa: E402
from contactscreen.rom import admissibility_indicator                            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fallback import N_CAL, RULES, SEED_RELS, pipeline_row, seed_mask        # noqa: E402

log = logging.getLogger("run_condensed")


def measure(problem, suite, condensed, models, mus, repeats: int) -> dict:
    nq, nr = len(mus), len(SEED_RELS)
    base = {k: np.zeros(nq) for k in ("t_suite", "it_suite", "t_cond", "it_cond", "t_back", "t_lcp", "du", "dlam")}
    data = {k: {"eta": np.zeros(nq), "err": np.zeros(nq), "conv": np.zeros(nq), "t_rom": np.zeros(nq),
                "t_warm": np.zeros((nr, nq)), "it_warm": np.zeros((nr, nq))} for k in models}
    for i, mu in enumerate(mus):
        f, c = problem.load(mu), problem.offset(mu)
        base["t_suite"][i], ref = time_call(lambda: suite.solve(mu), repeats)
        base["it_suite"][i] = ref.iterations
        base["t_cond"][i], cold = time_call(lambda: condensed.solve(f, c), repeats)
        base["it_cond"][i], base["t_back"][i], base["t_lcp"][i] = cold.iterations, cold.seconds_back_substitution, cold.seconds_lcp
        base["du"][i] = np.abs(cold.u - ref.u).max() / np.abs(ref.u).max()
        base["dlam"][i] = np.abs(cold.lam - ref.lam).max() / max(1e-300, np.abs(ref.lam).max())
        for key, model in models.items():
            t_rom, red = time_call(lambda: model.solve(mu, ref.K), repeats)
            eta, t_eta = admissibility_indicator(problem, mu, red)
            d = data[key]
            d["eta"][i], d["conv"][i], d["t_rom"][i] = eta, float(red.converged), t_rom + t_eta
            d["err"][i] = energy_error(ref.K, ref.u, red.u)
            for j, rel in enumerate(SEED_RELS):
                seed = seed_mask(red.lam, rel)
                d["t_warm"][j, i], warm = time_call(lambda: condensed.solve(f, c, active0=seed), repeats)
                d["it_warm"][j, i] = warm.iterations
        if (i + 1) % 10 == 0:
            log.info("%d/%d queries timed", i + 1, nq)
    return {"base": base, "models": data}


def summarise(m: dict, condensed, tune: slice, evl: slice, cal_audit, rng) -> dict:
    base, med = m["base"], lambda a: float(np.median(a[evl]))
    saving = float(np.mean(base["t_suite"][evl] - base["t_cond"][evl]))
    out = {"setup_ms": 1e3 * condensed.setup_seconds,
           "break_even_queries": float(condensed.setup_seconds / saving) if saving > 0 else None,
           "suite_ms_median": 1e3 * med(base["t_suite"]), "suite_iterations_median": med(base["it_suite"]),
           "condensed_ms_median": 1e3 * med(base["t_cond"]), "condensed_iterations_median": med(base["it_cond"]),
           "condensed_back_substitution_ms_median": 1e3 * med(base["t_back"]), "condensed_lcp_ms_median": 1e3 * med(base["t_lcp"]),
           "speedup_condensed_over_suite": float(base["t_suite"][evl].mean() / base["t_cond"][evl].mean()),
           "max_rel_du": float(base["du"][evl].max()), "max_rel_dlam": float(base["dlam"][evl].max()), "models": {}}
    t_cond = base["t_cond"][evl]
    for key, d in m["models"].items():
        g = lambda f: cal_audit.arrays[f"pool|{key[0]}|{key[1]}|{f}"]
        pool = labels(g("eta"), g("err"), g("converged"))
        idx = rng.permutation(len(pool.eta))[:N_CAL]
        cal = type(pool)(pool.eta[idx], pool.silent[idx], pool.good[idx], pool.converged[idx])
        test = labels(d["eta"][evl], d["err"][evl], d["conv"][evl])
        best = int(np.argmin(np.median(d["t_warm"][:, tune], axis=1)))
        t_warm = d["t_warm"][best][evl]
        entry = {"chosen_rel": SEED_RELS[best], "t_rom_ms_median": 1e3 * med(d["t_rom"]),
                 "t_warm_ms_median": 1e3 * float(np.median(t_warm)), "it_warm_median": float(np.median(d["it_warm"][best][evl])),
                 "t_cold_ms_median": 1e3 * float(np.median(t_cond)), "rules": {}}
        for rule, fn in RULES.items():
            tau, strict = fn(cal)
            entry["rules"][rule] = {"tau_defined": bool(np.isfinite(tau)),
                                    **pipeline_row(flagged(test, tau, strict), test, d, t_cond, t_warm, evl)}
        entry["rules"]["exact_error"] = {"tau_defined": True, **pipeline_row(~test.converged | test.silent, test, d, t_cond, t_warm, evl)}
        out["models"][f"{key[0]}|{key[1]}"] = entry
    return out


def print_tables(r: dict) -> None:
    s = r["H"]
    print(f"\n=== H ({r['problem']}, {r['n_eval']} evaluation queries) ===")
    print(f"set-up {s['setup_ms']:.0f} ms (break-even after {s['break_even_queries']:.0f} queries); suite cold {s['suite_ms_median']:.1f} ms "
          f"({s['suite_iterations_median']:.0f} it) vs condensed cold {s['condensed_ms_median']:.2f} ms ({s['condensed_iterations_median']:.0f} it; "
          f"back-substitution {s['condensed_back_substitution_ms_median']:.2f} ms, LCP {s['condensed_lcp_ms_median']:.2f} ms) "
          f"-> {s['speedup_condensed_over_suite']:.1f}x; agreement du {s['max_rel_du']:.1e}, dlam {s['max_rel_dlam']:.1e}")
    print("speedup of the screened pipeline against the condensed cold solve (cold | warm fallback), fallback share, missed/silent")
    for name, v in s["models"].items():
        print(f"{name}  (warm seed rel {v['chosen_rel']}: {v['t_warm_ms_median']:.2f} ms, ROM+indicator {v['t_rom_ms_median']:.2f} ms)")
        for rule, row in v["rules"].items():
            print(f"   {rule:<9}{row['speedup_cold']:>7.2f}{row['speedup_warm']:>7.2f}   fallback {row['fallback']:.2f}   "
                  f"missed {row['missed']}/{row['n_silent']}")


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
    problem, settings = make_problem(args.problem), make_settings(args.problem, args.seed)
    if problem.k_varies:
        raise ValueError(f"{args.problem} has a parameter-dependent stiffness; the condensed solver needs a constant K")
    path = audit_path(args.problem, args.seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/generate_audits.py first")
    suite = FullOrderSolver(problem)
    models = build_models(problem, suite, problem.sample(np.random.default_rng(settings.seed), settings.n_train), settings)
    condensed = CondensedSolver(problem.stiffness(problem.lo), problem.B)
    mus = problem.sample(np.random.default_rng(SEEDS["timing_queries"] + args.seed), args.n_time)
    log.info("%s: N=%d, n_c=%d, set-up %.0f ms, timing %d queries", problem.name, problem.n_dof, problem.n_contact,
             1e3 * condensed.setup_seconds, args.n_time)
    m = measure(problem, suite, condensed, models, mus, args.repeats)
    half = args.n_time // 2
    result = {"problem": args.problem, "n_time": args.n_time, "repeats": args.repeats, "n_eval": args.n_time - half,
              "seeds": SEEDS, "replicate": args.seed, "environment": environment_info(),
              "H": summarise(m, condensed, slice(0, half), slice(half, None), load_audit(path), np.random.default_rng(SEEDS["split"]))}
    out_path = RESULTS / f"condensed_{args.problem}.json"
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    log.info("wrote %s", out_path)
    print_tables(result)


if __name__ == "__main__":
    main()
