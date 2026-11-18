"""Split-conformal calibration of the silent-failure screen: feasibility check on the saved audits (B1, B2, B5).

Self-contained: reads the audit files written by the paper-1 project (``reduced-contact-reliability``), which stays
read-only. Nothing here imports the ``contact_rom`` package.

The calibrated screen flags a query when ``eta_adm > tau``. A miss is a silent failure with ``eta_adm <= tau``.
Rule compared with the first paper's ``0.9 * min`` rule:

*Conformal rule.* Let ``n`` be the number of silent failures among the calibration queries and ``k = floor(alpha (n+1))``.
``tau`` is the ``k``-th smallest indicator value of those silent failures (``tau = -inf``, flag every query, when
``k < 1``). If a new silent failure is exchangeable with the calibration silent failures, the probability of a miss is
``k / (n + 1) <= alpha``. The guarantee is marginal over the calibration set and class-conditional on the failures.

Protocol (as in the paper-1 calibration study): per seed (42-49) a pool of 200 validation queries and 100 test queries;
calibration sets are random subsets of the pool (20 draws per size, one draw for the full pool). Each (seed, basis,
``r_lambda``) entry is one run; counts are pooled over runs.

Run with the paper-1 interpreter (the audit files need its NumPy version), from this project folder:
``..\\reduced-contact-reliability\\.venv\\Scripts\\python.exe scripts\\conformal_screen.py``.
Writes ``results/conformal_screen.json``.
"""

import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
PAPER1 = HERE.parent / "reduced-contact-reliability"                  # read-only source of the audits
AUDITS = PAPER1 / "results" / "calibration"
OUT = HERE / "results" / "conformal_screen.json"

EPS, TOL = 0.05, 1e-12              # silent failure: reduced KKT residual <= TOL and relative energy error > EPS
BENCHES = ("b1", "b2", "b5")
SEEDS = tuple(range(42, 50))
KINDS = ("svd", "nmf", "cpg", "gis", "rid")
ALPHAS = (0.01, 0.05, 0.10)
N_CAL = (20, 40, 80, 200)
N_DRAWS = 20


def load(bench: str, seed: int) -> dict:
    with open(AUDITS / f"audits_{bench}_seed{seed}.pkl", "rb") as fh:
        return pickle.load(fh)


def conformal_tau(pos_values: np.ndarray, alpha: float) -> float:
    n = len(pos_values)
    k = int(np.floor(alpha * (n + 1)))
    return -np.inf if k < 1 else float(np.sort(pos_values)[k - 1])


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for bench in BENCHES:
        data = {s: load(bench, s) for s in SEEDS}
        for kind in KINDS:
            for n_cal in N_CAL:
                acc = {a: np.zeros(6) for a in ALPHAS}       # miss, positives, fp, negatives, flag_all_runs, runs
                base = np.zeros(6)                            # 0.9 min rule: miss, positives, fp, negatives, undefined, runs
                run_miss = {a: [] for a in ALPHAS}
                for s, d in data.items():
                    for (k, r), val in d["val"].items():
                        if k != kind:
                            continue
                        test = d["test"][(k, r)]
                        pool = len(val["eta_adm"])
                        pos_t = (test["res_red"] <= TOL) & (test["err_energy"] > EPS)
                        neg_t = test["err_energy"] <= EPS
                        eta_t = test["eta_adm"]
                        silent_v = (val["res_red"] <= TOL) & (val["err_energy"] > EPS)
                        draws = 1 if n_cal >= pool else N_DRAWS
                        for _ in range(draws):
                            idx = np.arange(pool) if n_cal >= pool else rng.choice(pool, n_cal, replace=False)
                            pv = val["eta_adm"][idx][silent_v[idx]]
                            for a in ALPHAS:
                                tau = conformal_tau(pv, a)
                                flag = eta_t > tau
                                m = int((pos_t & ~flag).sum())
                                acc[a] += [m, pos_t.sum(), (neg_t & flag).sum(), neg_t.sum(), float(np.isinf(tau)), 1]
                                if pos_t.any():
                                    run_miss[a].append(m / pos_t.sum())
                            if len(pv):
                                flag = eta_t > 0.9 * pv.min()
                                base += [(pos_t & ~flag).sum(), pos_t.sum(), (neg_t & flag).sum(), neg_t.sum(), 0, 1]
                            else:
                                base += [0, 0, 0, 0, 1, 1]
                for a in ALPHAS:
                    m, p, fp, ng, fa, runs = acc[a]
                    out[f"{bench}|{kind}|{n_cal}|{a}"] = {
                        "miss_pooled": m / p if p else None, "miss_run_mean": float(np.mean(run_miss[a])) if run_miss[a] else None,
                        "fpr": fp / ng if ng else None, "flag_all_share": fa / runs, "n_pos": int(p), "runs": int(runs)}
                m, p, fp, ng, und, runs = base
                out[f"{bench}|{kind}|{n_cal}|min09"] = {"miss_pooled": m / p if p else None, "fpr": fp / ng if ng else None,
                                                         "undefined_share": und / runs, "n_pos": int(p), "runs": int(runs)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda x: "  --" if x is None else f"{x:5.3f}"
    for bench in BENCHES:
        print(f"\n===== {bench.upper()}  (40 calibration queries): miss rate | FPR | flag-all share")
        print(f"{'basis':<5}" + "".join(f"  a={a:<4} miss   fpr  flagall" for a in ALPHAS) + "   min09 miss   fpr  undef")
        for kind in KINDS:
            row = f"{kind:<5}"
            for a in ALPHAS:
                r = out[f"{bench}|{kind}|40|{a}"]
                row += f"  {f(r['miss_pooled'])} {f(r['fpr'])} {r['flag_all_share']:5.2f}   "
            b = out[f"{bench}|{kind}|40|min09"]
            row += f"   {f(b['miss_pooled'])} {f(b['fpr'])} {b['undefined_share']:5.2f}"
            print(row)


if __name__ == "__main__":
    main()
