"""Experiment B (calibration size needed) and the B4 hold-out check for the conformal silent-failure screen.

Reuses the rule and the audit loader of ``conformal_screen.py`` (same protocol: seeds 42-49, pool of 200 validation
queries and 100 test queries per run, random calibration subsets, 20 draws per size). Reads the saved paper-1 audits,
which stay read-only. Does not import the ``contact_rom`` package.

Experiment B. With ``n`` silent failures among the calibration queries the rule needs ``k = floor(alpha (n+1)) >= 1``,
i.e. ``n >= ceil(1/alpha) - 1``; otherwise it flags every query. Given the silent-failure rate ``p`` of a basis, about
``n_min / p`` calibration queries are needed. The script reports ``p`` (seed-level bootstrap interval), the implied
calibration size, and the share of runs in which the rule is defined (1 - flag-all share) for each calibration size.

B4 hold-out. B4 (single 3D block, moving contact patch) was not used to design the rule. The script checks whether the
pooled miss rate stays at or below ``alpha`` there (seed-level bootstrap interval), alongside the false-alarm rate.

Run with the paper-1 interpreter, from this project folder:
``..\\reduced-contact-reliability\\.venv\\Scripts\\python.exe scripts\\conformal_size_holdout.py``.
Writes ``results/conformal_size_holdout.json``.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np

from conformal_screen import (ALPHAS, BENCHES, EPS, KINDS, N_CAL, N_DRAWS, SEEDS, TOL, AUDITS, conformal_tau, load)

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "conformal_size_holdout.json"

HOLDOUT = "b4"
ALL_BENCHES = BENCHES + (HOLDOUT,)
N_BOOT = 2000                    # seed-level bootstrap resamples
BOOT_SEED = 1
CI = (2.5, 97.5)
COLS = ("miss", "pos", "fp", "neg", "flag_all", "runs")

log = logging.getLogger("conformal_size_holdout")


def n_min(alpha: float) -> int:
    """Smallest number of calibration silent failures for which the rule is defined (k >= 1)."""
    return math.ceil(1.0 / alpha) - 1


def validate_inputs() -> None:
    missing = [f"audits_{b}_seed{s}.pkl" for b in ALL_BENCHES for s in SEEDS if not (AUDITS / f"audits_{b}_seed{s}.pkl").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audit files missing in {AUDITS}, first: {missing[0]}")


def run_cells(data: dict, bench: str, kind: str, n_cal: int, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """Per-seed counts for each alpha, and per-seed (silent failures, queries) of the validation pools of one basis."""
    cells = {a: np.zeros((len(SEEDS), len(COLS))) for a in ALPHAS}
    pool_counts = np.zeros((len(SEEDS), 2))
    for i, seed in enumerate(SEEDS):
        d = data[seed]
        for (k, r), val in d["val"].items():
            if k != kind:
                continue
            test = d["test"][(k, r)]
            pool = len(val["eta_adm"])
            silent_v = (val["res_red"] <= TOL) & (val["err_energy"] > EPS)
            pos_t = (test["res_red"] <= TOL) & (test["err_energy"] > EPS)
            neg_t = test["err_energy"] <= EPS
            eta_t = test["eta_adm"]
            pool_counts[i] += [silent_v.sum(), pool]
            for _ in range(1 if n_cal >= pool else N_DRAWS):
                idx = np.arange(pool) if n_cal >= pool else rng.choice(pool, n_cal, replace=False)
                pos_values = val["eta_adm"][idx][silent_v[idx]]
                for a in ALPHAS:
                    tau = conformal_tau(pos_values, a)
                    flag = eta_t > tau
                    cells[a][i] += [(pos_t & ~flag).sum(), pos_t.sum(), (neg_t & flag).sum(), neg_t.sum(), float(np.isinf(tau)), 1]
    return cells, pool_counts


def pooled_ratio(per_seed: np.ndarray, num: int, den: int) -> float | None:
    total_den = per_seed[:, den].sum()
    return float(per_seed[:, num].sum() / total_den) if total_den else None


def bootstrap_ci(per_seed: np.ndarray, num: int, den: int, rng: np.random.Generator) -> list | None:
    """Seed-level bootstrap interval of sum(num)/sum(den); None when the denominator is empty."""
    if per_seed[:, den].sum() == 0:
        return None
    draws = rng.integers(0, len(per_seed), size=(N_BOOT, len(per_seed)))
    sums_num, sums_den = per_seed[draws, num].sum(axis=1), per_seed[draws, den].sum(axis=1)
    ok = sums_den > 0
    return [float(x) for x in np.percentile(sums_num[ok] / sums_den[ok], CI)]


def size_needed(pool_counts: np.ndarray, rng: np.random.Generator) -> dict:
    """Silent-failure rate p of a basis and the calibration size n_min / p it implies, per alpha."""
    p = pooled_ratio(pool_counts, 0, 1)
    p_ci = bootstrap_ci(pool_counts, 0, 1, rng)
    out = {"p_silent": p, "p_silent_ci": p_ci, "n_silent": int(pool_counts[:, 0].sum()), "n_queries": int(pool_counts[:, 1].sum())}
    for a in ALPHAS:
        needs = lambda q: math.ceil(n_min(a) / q) if q else None          # q = rate; None when the rate is zero
        out[f"alpha={a}"] = {"n_min_silent": n_min(a), "n_queries_needed": needs(p),
                             "n_queries_needed_range": [needs(p_ci[1]), needs(p_ci[0])] if p_ci else None}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_inputs()
    rng = np.random.default_rng(0)
    boot_rng = np.random.default_rng(BOOT_SEED)
    out = {"cells": {}, "rates": {}, "protocol": {"seeds": list(SEEDS), "n_cal": list(N_CAL), "alphas": list(ALPHAS),
                                                   "draws": N_DRAWS, "holdout": HOLDOUT, "n_boot": N_BOOT}}
    for bench in ALL_BENCHES:
        log.info("benchmark %s", bench)
        data = {s: load(bench, s) for s in SEEDS}
        for kind in KINDS:
            for n_cal in N_CAL:
                cells, pool_counts = run_cells(data, bench, kind, n_cal, rng)
                for a in ALPHAS:
                    c = cells[a]
                    out["cells"][f"{bench}|{kind}|{n_cal}|{a}"] = {
                        "miss_pooled": pooled_ratio(c, 0, 1), "miss_ci": bootstrap_ci(c, 0, 1, boot_rng),
                        "fpr": pooled_ratio(c, 2, 3), "defined_share": 1.0 - c[:, 4].sum() / c[:, 5].sum(),
                        "n_pos": int(c[:, 1].sum()), "runs": int(c[:, 5].sum())}
            out["rates"][f"{bench}|{kind}"] = size_needed(pool_counts, boot_rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT)
    print_tables(out)


def print_tables(out: dict) -> None:
    f = lambda x, w=5: " " * (w - 2) + "--" if x is None else f"{x:{w}.3f}"
    print("\n=== B: silent-failure rate p and calibration queries needed (alpha = 0.05 | 0.10)")
    print(f"{'bench':<6}{'basis':<6}{'p':>7}   {'p 95% CI':<16}{'need@0.05':>10}{'need@0.10':>10}")
    for key, r in out["rates"].items():
        bench, kind = key.split("|")
        ci = "--" if r["p_silent_ci"] is None else f"[{r['p_silent_ci'][0]:.3f},{r['p_silent_ci'][1]:.3f}]"
        need = lambda a: r[f"alpha={a}"]["n_queries_needed"]
        print(f"{bench:<6}{kind:<6}{f(r['p_silent'], 7)}   {ci:<16}{str(need(0.05)):>10}{str(need(0.1)):>10}")
    print(f"\n=== B: share of runs with a defined rule (alpha = 0.05), by calibration size {tuple(N_CAL)}")
    for bench in ALL_BENCHES:
        for kind in KINDS:
            shares = [out["cells"][f"{bench}|{kind}|{n}|0.05"]["defined_share"] for n in N_CAL]
            print(f"{bench:<6}{kind:<6}" + "".join(f(s, 8) for s in shares))
    print(f"\n=== B4 hold-out: pooled miss rate [95% CI] | FPR | defined share (200 and 80 calibration queries)")
    for a in (0.05, 0.10):
        for n in (200, 80):
            print(f"-- alpha={a}, n_cal={n}")
            for kind in KINDS:
                c = out["cells"][f"{HOLDOUT}|{kind}|{n}|{a}"]
                ci = "--" if c["miss_ci"] is None else f"[{c['miss_ci'][0]:.3f},{c['miss_ci'][1]:.3f}]"
                print(f"  {kind:<5}miss {f(c['miss_pooled'])} {ci:<16} fpr {f(c['fpr'])}  defined {c['defined_share']:.2f}  n_pos {c['n_pos']}")


if __name__ == "__main__":
    main()
