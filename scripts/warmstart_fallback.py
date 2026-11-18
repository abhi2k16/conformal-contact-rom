"""Experiments D and E: warm-started fallback and net speedup with the conformal screen (B2 and B3).

Reads the paper-1 package ``contact_rom`` through a path setting (it is imported, not copied, and stays read-only).
Run with the paper-1 interpreter, one process at a time (timings), from this project folder:

    ..\\reduced-contact-reliability\\.venv\\Scripts\\python.exe scripts\\warmstart_fallback.py --bench b2
    ..\\reduced-contact-reliability\\.venv\\Scripts\\python.exe scripts\\warmstart_fallback.py --bench b3

Experiment D (cost of one fallback solve). The full-order solve is a linear complementarity problem (LCP) in the
contact multipliers, ``w = A z + q``, with ``A = B K^-1 B^T``. Three solvers are compared on the same queries:
  * cold:  ``lcp_active_set(A, q)``, the paper-1 fallback;
  * warm:  ``lcp_active_set(A, q, active0=S)`` with ``S`` the positive multipliers of the lifted reduced solution (H4);
  * lazy:  the same primal-dual iteration as ``warm``, but it forms only the columns of ``A`` (and ``K^-1 B^T``) that
           the active set needs, so a good seed also avoids most of the ``n_c`` triangular solves. B3 only
           (parameter-varying ``K``), because there the compliance assembly dominates the fallback cost.
The warm and lazy answers are checked against the cold one.

Experiment E (net speedup, B3). Per basis and dual dimension, the screen flags a query when ``eta_adm > tau`` with
``tau`` from the conformal rule (alpha 0.05 and 0.10) or the paper-1 ``0.9 min`` rule, calibrated on the validation
queries. A flagged query pays the reduced attempt plus a fallback solve (cold, warm or lazy). The speedup is the
cold full-order time over the mean cost per query, as in paper 1.

Raw per-query data go to ``warmstart_<bench>.pkl``, the summary to ``warmstart_<bench>.json`` (``--out-dir``,
default ``results/``). ``--analyse-only`` recomputes the summary from the pickle.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread, as the paper-1 timings
    os.environ[_var] = "1"

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parents[1]
PAPER1 = HERE.parent / "reduced-contact-reliability"
sys.path.insert(0, str(PAPER1))                      # read-only import of the paper-1 package

from conformal_screen import EPS, KINDS, TOL, conformal_tau                      # noqa: E402
from contact_rom.audit import (build_reduced_model, evaluate_query, reduced_factors, rom_for_query,   # noqa: E402
                               solve_reduced_timed)
from contact_rom.bases import build_snapshots, pod_primal_basis                    # noqa: E402
from contact_rom.benchmark_3d import build_b3                                      # noqa: E402
from contact_rom.benchmarks import build_b2                                        # noqa: E402
from contact_rom.config import Config                                              # noqa: E402
from contact_rom.query_context import QueryContext, solve_fom_factored             # noqa: E402
from rom_engine.complementarity import lcp_active_set                              # noqa: E402

DEFAULT_OUT = HERE / "results"
R_DUAL = (8, 16)
ALPHAS = (0.05, 0.10)
ACT_TOL = 1e-12                  # multiplier counted as active (as in paper 1)
KEEP = ("res_red", "err_energy", "eta_adm", "t_online", "m_r_act")    # audit fields stored per query
LAZY_TOL = 1e-8                  # accepted difference between warm/lazy and cold answers (relative to max |z|)

log = logging.getLogger("warmstart_fallback")


# ----------------------------------------------------------------------------------------------- setup

def build_setup(bench: str, cfg: Config, n_val: int, n_test: int) -> tuple:
    """Problem, training snapshots and the validation / test query sets (one seeded generator)."""
    problem = build_b2(cfg.b2) if bench == "b2" else build_b3(cfg.b3)
    rng = np.random.default_rng(cfg.seed + 1)
    n_train = cfg.n_train if bench == "b2" else cfg.e5_n_train
    snaps = build_snapshots(problem, problem.sample_mu(rng, n_train), cfg.lcp_method)
    mu_sets = {"test": problem.sample_mu(rng, n_test), "val": problem.sample_mu(rng, n_val)}
    return problem, snaps, mu_sets


def build_models(problem, cfg: Config, snaps, kinds: tuple) -> tuple:
    """``{(kind, r): (rom, meta)}``, the offline factors (None when K varies) and ``C_lambda``."""
    v_pod = pod_primal_basis(snaps.U_snap, cfg.r_primal)
    c_lambda = float(np.linalg.norm(snaps.Lam_snap, axis=0).max())
    models = {(k, r): build_reduced_model(problem, cfg, snaps, v_pod, k, r, repair=True) for k in kinds for r in R_DUAL}
    factors = {key: None if problem.k_varies else reduced_factors(m[0]) for key, m in models.items()}
    return models, factors, c_lambda


def fom_context(problem, mu: np.ndarray) -> tuple:
    """Paper-1 query context plus the LCP data ``(lu, A, q)`` that its builder discards."""
    sol, lu, a_lcp, iters = solve_fom_factored(problem, mu)
    assert sol.converged, "FOM LCP did not converge"
    k_mat, b_mat = problem.K_mat(mu), problem.B_mat(mu)
    f_vec, c_vec = problem.f_vec(mu), problem.c_vec(mu)
    b_norm = float(np.sqrt(max(np.linalg.eigvalsh(a_lcp)[-1], 0.0)))
    ctx = QueryContext(mu, k_mat, b_mat, f_vec, c_vec, lu, b_norm, sol, iters)
    q_vec = -(b_mat @ lu.solve(f_vec)) + c_vec
    return ctx, a_lcp, q_vec


# ------------------------------------------------------------------------------------------ solvers, timing

def best_of(func, repeats: int) -> tuple:
    """``(minimum wall time, last result)`` of ``repeats`` calls."""
    best, out = np.inf, None
    for _ in range(repeats):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return float(best), out


def lazy_active_set(lu, bs_mat: sp.spmatrix, q_vec: np.ndarray, active0, tol: float = 1e-12, max_iter: int = 100):
    """Primal-dual active-set iteration of ``lcp_active_set`` with lazily formed columns of ``A = Bs K^-1 Bs^T``.

    Returns ``(z, iterations, X)`` where ``X[j] = K^-1 Bs[j]^T`` for the active columns, or ``None`` if the active
    sets cycle (the caller then uses the cold solver). The iterates equal those of ``lcp_active_set(active0=...)``.
    """
    n = len(q_vec)
    cols, x_cols, seen = {}, {}, []
    active = {int(i) for i in active0}
    for it in range(1, max_iter + 1):
        idx = np.array(sorted(active), dtype=int)
        z, w = np.zeros(n), q_vec.copy()
        if idx.size:
            need = [j for j in idx if j not in cols]
            if need:
                x_new = lu.solve(bs_mat[need].T.toarray())
                a_new = bs_mat @ x_new
                for k, j in enumerate(need):
                    x_cols[j], cols[j] = x_new[:, k], a_new[:, k]
            a_cols = np.column_stack([cols[j] for j in idx])
            a_ss = 0.5 * (a_cols[idx] + a_cols[idx].T)
            z[idx] = np.linalg.solve(a_ss, -q_vec[idx])
            w = a_cols @ z[idx] + q_vec
        new = set(np.flatnonzero(z - w > tol).tolist())
        if new == active and np.all(z >= -1e-10) and np.all(w >= -1e-10):
            return np.maximum(z, 0.0), it, x_cols
        if frozenset(new) in seen:
            return None
        seen.append(frozenset(active))
        active = new
    return None


def time_cold_parts(problem, mu: np.ndarray, repeats: int) -> dict:
    """Wall time of the cold fallback split into factorisation, right-hand side, compliance, and LCP (K varying)."""
    k_mat, bs = sp.csc_matrix(problem.K_mat(mu)), sp.csr_matrix(-problem.B_mat(mu))
    f_vec, c_vec = problem.f_vec(mu), -problem.c_vec(mu)
    t_lu, lu = best_of(lambda: spla.splu(k_mat), repeats)
    t_rhs, kinv_f = best_of(lambda: lu.solve(f_vec), repeats)
    t_comp, a_lcp = best_of(lambda: _compliance(lu, bs), repeats)
    q_vec = bs @ kinv_f - c_vec
    t_lcp, _ = best_of(lambda: lcp_active_set(a_lcp, q_vec), repeats)
    return {"t_lu": t_lu, "t_rhs": t_rhs, "t_compliance": t_comp, "t_lcp_cold": t_lcp, "lu": lu, "bs": bs, "q": q_vec,
            "kinv_f": kinv_f, "a_lcp": a_lcp}


def _compliance(lu, bs: sp.spmatrix) -> np.ndarray:
    a_mat = bs @ lu.solve(bs.T.toarray())
    return 0.5 * (a_mat + a_mat.T)


def measure_warm(a_lcp: np.ndarray, q_vec: np.ndarray, z_cold: np.ndarray, active0: np.ndarray, repeats: int) -> dict:
    """Warm-started LCP (D, ``warm``): iterations, wall time and difference from the cold answer."""
    t_warm, (z, ok, iters) = best_of(lambda: lcp_active_set(a_lcp, q_vec, active0=active0), repeats)
    return {"it_warm": int(iters), "t_warm": t_warm, "ok": bool(ok),
            "dz": float(np.max(np.abs(z - z_cold)) / max(1.0, np.max(np.abs(z_cold))))}


def measure_lazy(parts: dict, z_cold: np.ndarray, u_ref: np.ndarray, active0: np.ndarray, repeats: int) -> dict:
    """Lazy solve (D, ``lazy``): time of the solve incl. recovering ``u``, columns formed, deviation from cold."""
    def run():
        res = lazy_active_set(parts["lu"], parts["bs"], parts["q"], active0)
        if res is None:
            return None
        z, iters, x_cols = res
        u = parts["kinv_f"] + sum((x_cols[j] * z[j] for j in x_cols), np.zeros_like(parts["kinv_f"]))
        return z, iters, len(x_cols), u
    t_lazy, out = best_of(run, repeats)
    if out is None:
        return {"lazy_cycled": True, "t_lazy": t_lazy}
    z, iters, n_cols, u = out
    return {"lazy_cycled": False, "t_lazy": t_lazy, "it_lazy": int(iters), "n_cols": int(n_cols),
            "dz": float(np.max(np.abs(z - z_cold)) / max(1.0, np.max(np.abs(z_cold)))),
            "du": float(np.max(np.abs(u - u_ref)) / max(1.0, np.max(np.abs(u_ref))))}


# ------------------------------------------------------------------------------------------------ collection

def lifted_active_set(rom_res) -> np.ndarray:
    return np.flatnonzero(rom_res.lam > ACT_TOL)


def collect_query(problem, cfg, models, factors, c_lambda, mu, rows, d1, d2, fom_rows, do_d, n_repeat, timed):
    """Audit one query for every model; for test queries also time the fallback solvers (D)."""
    ctx, a_lcp, q_vec = fom_context(problem, mu)
    z_cold = ctx.fom.lam_vec
    s_fom = set(np.flatnonzero(z_cold > ACT_TOL).tolist())
    parts = time_cold_parts(problem, mu, n_repeat) if (timed and problem.k_varies) else None
    if do_d:
        t_cold, _ = best_of(lambda: lcp_active_set(a_lcp, q_vec), 3)
        fom_rows.append({"it_cold": ctx.fom_iterations, "t_lcp_cold": t_cold, "m_fom": len(s_fom),
                         **({k: parts[k] for k in ("t_lu", "t_rhs", "t_compliance")} if parts else {})})
    for key, (rom, meta) in models.items():
        rom_q = rom_for_query(problem, cfg, rom, ctx)
        metrics = evaluate_query(ctx, rom_q, c_lambda, factors[key])
        rows[key].append({k: metrics[k] for k in KEEP})
        if not do_d:
            continue
        res = solve_reduced_timed(rom_q, ctx.f_vec, -ctx.c_vec, factors[key])
        active0 = lifted_active_set(res)
        s_rom = set(active0.tolist())
        entry = measure_warm(a_lcp, q_vec, z_cold, active0, 3)
        entry["jaccard"] = len(s_rom & s_fom) / max(1, len(s_rom | s_fom))
        entry["m_rom"] = len(s_rom)
        d1[key].append(entry)
        if parts is not None:
            d2[key].append(measure_lazy(parts, z_cold, ctx.fom.u_vec, active0, n_repeat))


def collect(bench: str, cfg: Config, n_val: int, n_test: int, n_timed: int, n_repeat: int) -> dict:
    """All per-query data of one benchmark."""
    problem, snaps, mu_sets = build_setup(bench, cfg, n_val, n_test)
    models, factors, c_lambda = build_models(problem, cfg, snaps, KINDS)
    log.info("%s: N=%d, n_c=%d, %d models", bench, problem.K_mat(snaps.mu_set[0]).shape[0],
             problem.B_mat(snaps.mu_set[0]).shape[0], len(models))
    raw = {"bench": bench, "n_timed": n_timed, "models_meta": {k: {"t_proj": m[1]["t_proj"], "t_eta": m[1]["t_eta"]}
                                                                for k, m in models.items()}, "sets": {}}
    for name in ("test", "val"):                                   # test first: the timed queries come early
        rows = {k: [] for k in models}
        d1, d2 = {k: [] for k in models}, {k: [] for k in models}
        fom_rows: list = []
        t_start = time.perf_counter()
        for i, mu in enumerate(mu_sets[name]):
            collect_query(problem, cfg, models, factors, c_lambda, mu, rows, d1, d2, fom_rows,
                          do_d=(name == "test"), n_repeat=n_repeat, timed=(name == "test" and i < n_timed))
            if (i + 1) % 10 == 0:
                log.info("%s %s: %d/%d queries, %.0f s", bench, name, i + 1, len(mu_sets[name]),
                         time.perf_counter() - t_start)
        raw["sets"][name] = {"rows": rows, "d1": d1, "d2": d2, "fom": fom_rows}
    return raw


# ------------------------------------------------------------------------------------------------ analysis

def col(rows: list, key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], float)


def silent(rows: list) -> np.ndarray:
    return (col(rows, "res_red") <= TOL) & (col(rows, "err_energy") > EPS)


def summarise_d(raw: dict) -> dict:
    """Experiment D rows per (kind, r): iterations, LCP time, active-set overlap, and (B3) full fallback time."""
    test = raw["sets"]["test"]
    fom = test["fom"]
    out = {"fom": {"it_cold_median": float(np.median(col(fom, "it_cold"))),
                   "t_lcp_cold_ms_median": 1e3 * float(np.median(col(fom, "t_lcp_cold"))),
                   "m_fom_median": float(np.median(col(fom, "m_fom")))}, "models": {}}
    timed = [r for r in fom if "t_lu" in r]
    if timed:
        out["fom"].update({f"{k}_s_median": float(np.median(col(timed, k))) for k in ("t_lu", "t_rhs", "t_compliance")})
        out["fom"]["n_timed"] = len(timed)
    for key, d1 in test["d1"].items():
        n = len(d1)
        it_cold = col(fom, "it_cold")[:n]
        it_w = col(d1, "it_warm")
        row = {"it_warm_median": float(np.median(it_w)), "it_cold_median": float(np.median(it_cold)),
               "share_warm_fewer": float(np.mean(it_w < it_cold)), "share_warm_more": float(np.mean(it_w > it_cold)),
               "t_warm_ms_median": 1e3 * float(np.median(col(d1, "t_warm"))),
               "t_cold_ms_median": 1e3 * float(np.median(col(fom, "t_lcp_cold"))),
               "jaccard_median": float(np.median(col(d1, "jaccard"))), "max_dz": float(col(d1, "dz").max()),
               "all_ok": bool(all(r["ok"] for r in d1))}
        d2 = test["d2"][key]
        if d2:
            good = [r for r in d2 if not r["lazy_cycled"]]
            row["lazy"] = {"n": len(d2), "n_cycled": len(d2) - len(good)}
            if good:
                row["lazy"].update({"n_cols_median": float(np.median(col(good, "n_cols"))),
                                    "it_median": float(np.median(col(good, "it_lazy"))),
                                    "max_dz": float(col(good, "dz").max()), "max_du": float(col(good, "du").max()),
                                    "t_lazy_s_median": float(np.median(col(good, "t_lazy")))})
        out["models"][f"{key[0]}|{key[1]}"] = row
    return out


def fallback_times(raw: dict, key: tuple) -> dict | None:
    """Median fallback time (s) per variant for one model: cold, warm (LCP only), lazy. B3 only."""
    test = raw["sets"]["test"]
    timed = [r for r in test["fom"] if "t_lu" in r]
    if not timed:
        return None
    base = float(np.median([r["t_lu"] + r["t_rhs"] + r["t_compliance"] for r in timed]))
    shared = float(np.median([r["t_lu"] + r["t_rhs"] for r in timed]))
    cold = base + float(np.median(col(test["fom"], "t_lcp_cold")))
    warm = base + float(np.median(col(test["d1"][key], "t_warm")))
    good = [r for r in test["d2"][key] if not r["lazy_cycled"]]
    lazy = shared + float(np.median(col(good, "t_lazy"))) if good else None
    return {"cold": cold, "warm": warm, "lazy": lazy}


def rule_tau(rows_val: list, rule: str) -> float:
    pos = col(rows_val, "eta_adm")[silent(rows_val)]
    if rule == "min09":
        return 0.9 * float(pos.min()) if len(pos) else -np.inf
    return conformal_tau(pos, float(rule.split("=")[1]))


def summarise_e(raw: dict) -> dict:
    """Experiment E rows per (kind, r, rule): fallback share, missed silent failures, speedup per fallback variant."""
    val, test = raw["sets"]["val"], raw["sets"]["test"]
    out = {}
    for key in test["rows"]:
        times = fallback_times(raw, key)
        if times is None:
            continue
        t_rom = np.array([raw["models_meta"][key]["t_proj"] + raw["models_meta"][key]["t_eta"] + r["t_online"]
                          for r in test["rows"][key]])
        eta, sil = col(test["rows"][key], "eta_adm"), silent(test["rows"][key])
        n_sil_val = int(silent(val["rows"][key]).sum())
        for rule in [f"alpha={a}" for a in ALPHAS] + ["min09"]:
            tau = rule_tau(val["rows"][key], rule)
            flagged = eta > tau
            row = {"tau_defined": bool(np.isfinite(tau)), "val_silent": n_sil_val, "test_silent": int(sil.sum()),
                   "fallback": float(flagged.mean()), "missed": int((sil & ~flagged).sum())}
            for variant, t_fb in times.items():
                row[f"speedup_{variant}"] = None if t_fb is None else float(times["cold"] / np.mean(t_rom + flagged * t_fb))
            out[f"{key[0]}|{key[1]}|{rule}"] = row
        out[f"{key[0]}|{key[1]}|times"] = {**times, "t_rom_mean": float(t_rom.mean())}
    return out


def print_tables(summary: dict, bench: str) -> None:
    f = lambda x, p=2: "  --" if x is None else f"{x:.{p}f}"
    d = summary["D"]
    print(f"\n=== D ({bench.upper()}): cold LCP median {d['fom']['it_cold_median']:.0f} iterations, "
          f"{d['fom']['t_lcp_cold_ms_median']:.2f} ms; FOM active set {d['fom']['m_fom_median']:.0f}")
    for k in ("t_lu", "t_rhs", "t_compliance"):
        if f"{k}_s_median" in d["fom"]:
            print(f"    {k}: {d['fom'][k + '_s_median']:.2f} s (median of {d['fom']['n_timed']} timed queries)")
    print(f"{'basis':<5}{'r':>3}{'it cold':>9}{'it warm':>9}{'fewer':>7}{'more':>6}{'ms cold':>9}{'ms warm':>9}{'Jaccard':>9}"
          f"{'max dz':>10}{'lazy cols':>10}{'lazy s':>8}")
    for name, r in d["models"].items():
        kind, rr = name.split("|")
        lz = r.get("lazy", {})
        print(f"{kind:<5}{rr:>3}{r['it_cold_median']:>9.1f}{r['it_warm_median']:>9.1f}{r['share_warm_fewer']:>7.2f}"
              f"{r['share_warm_more']:>6.2f}{r['t_cold_ms_median']:>9.2f}{r['t_warm_ms_median']:>9.2f}"
              f"{r['jaccard_median']:>9.2f}{r['max_dz']:>10.1e}{f(lz.get('n_cols_median'), 0):>10}{f(lz.get('t_lazy_s_median')):>8}")
    e = summary.get("E")
    if not e:
        return
    print(f"\n=== E ({bench.upper()}): net speedup with cold | warm | lazy fallback; fallback share; missed/test silent")
    print(f"{'basis':<5}{'r':>3} {'rule':<12}{'val sil':>8}{'fallback':>9}{'missed':>8}{'cold':>7}{'warm':>7}{'lazy':>7}")
    for name, r in e.items():
        parts = name.split("|")
        if parts[2] == "times":
            continue
        print(f"{parts[0]:<5}{parts[1]:>3} {parts[2]:<12}{r['val_silent']:>8d}{r['fallback']:>9.2f}"
              f"{r['missed']:>4d}/{r['test_silent']:<3d}{f(r['speedup_cold']):>7}{f(r['speedup_warm']):>7}{f(r['speedup_lazy']):>7}")


def analyse(raw: dict) -> dict:
    return {"D": summarise_d(raw), "E": summarise_e(raw), "bench": raw["bench"]}


# ------------------------------------------------------------------------------------------------ entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", choices=("b2", "b3"), required=True)
    p.add_argument("--n-val", type=int, default=200, help="validation queries (calibration pool)")
    p.add_argument("--n-test", type=int, default=100, help="test queries")
    p.add_argument("--n-timed", type=int, default=10, help="B3: test queries for which the fallback parts are timed")
    p.add_argument("--repeats", type=int, default=2, help="best-of repeats for the timed fallback parts")
    p.add_argument("--b3-n-xy", type=int, default=None, help="override B3 elements per side (development)")
    p.add_argument("--b3-nz", type=int, default=None, help="override B3 layers per body (development)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--analyse-only", action="store_true", help="recompute the summary from the saved pickle")
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not PAPER1.exists():
        raise FileNotFoundError(f"paper-1 project not found at {PAPER1}")
    if args.n_val < 1 or args.n_test < 1 or args.n_timed < 0 or args.repeats < 1:
        raise ValueError("--n-val, --n-test, --repeats must be positive and --n-timed non-negative")
    if args.bench == "b2" and (args.b3_n_xy or args.b3_nz):
        raise ValueError("--b3-* options apply to --bench b3 only")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    validate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pkl, js = args.out_dir / f"warmstart_{args.bench}.pkl", args.out_dir / f"warmstart_{args.bench}.json"
    if args.analyse_only:
        with open(pkl, "rb") as fh:
            raw = pickle.load(fh)
    else:
        cfg = Config()
        if args.bench == "b3":
            b3 = replace(cfg.b3, n_xy=args.b3_n_xy or cfg.b3.n_xy, nz_up=args.b3_nz or cfg.b3.nz_up,
                         nz_low=args.b3_nz or cfg.b3.nz_low)
            cfg = replace(cfg, b3=b3)
        raw = collect(args.bench, cfg, args.n_val, args.n_test, args.n_timed, args.repeats)
        with open(pkl, "wb") as fh:
            pickle.dump(raw, fh)
        log.info("wrote %s", pkl)
    summary = analyse(raw)
    js.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    log.info("wrote %s", js)
    print_tables(summary, args.bench)


if __name__ == "__main__":
    main()
