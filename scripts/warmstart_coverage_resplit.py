"""Coverage check of the conformal screen on B3 by random re-splitting of the saved queries.

``warmstart_fallback.py`` calibrates on one fixed validation set (200 queries) and scores one fixed test set (100
queries), so its miss rates carry the variance of a single calibration draw. The validation and test queries are
drawn from the same distribution, hence the 300 queries are exchangeable. This script pools them, draws many random
200/100 splits, calibrates on the first part, scores the second, and reports the miss rate averaged over splits.

Input: ``results/warmstart_b3.pkl`` (written by ``warmstart_fallback.py``). Output: ``results/warmstart_b3_resplit.json``.
Run with the paper-1 interpreter, from this project folder (the pickle holds only NumPy data and plain dicts):
``..\\reduced-contact-reliability\\.venv\\Scripts\\python.exe scripts\\warmstart_coverage_resplit.py``.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from conformal_screen import EPS, TOL, conformal_tau

HERE = Path(__file__).resolve().parents[1]
PICKLE = HERE / "results" / "warmstart_b3.pkl"
OUT = HERE / "results" / "warmstart_b3_resplit.json"
ALPHAS = (0.05, 0.10)
N_SPLITS = 500
SEED = 0

log = logging.getLogger("warmstart_coverage_resplit")


def query_arrays(rows: list) -> tuple:
    """Indicator values and silent-failure mask of one model's queries."""
    eta = np.array([r["eta_adm"] for r in rows], float)
    silent = np.array([(r["res_red"] <= TOL) and (r["err_energy"] > EPS) for r in rows])
    return eta, silent


def resplit_model(eta: np.ndarray, silent: np.ndarray, n_cal: int, rng: np.random.Generator) -> dict:
    """Per alpha: miss rate pooled over splits, its spread across splits, and the share of splits with a flag-all rule."""
    out = {a: {"missed": 0, "silent": 0, "split_rates": [], "flag_all": 0} for a in ALPHAS}
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(eta))
        cal, test = perm[:n_cal], perm[n_cal:]
        pos_cal = eta[cal][silent[cal]]
        for a in ALPHAS:
            tau = conformal_tau(pos_cal, a)
            miss = int((silent[test] & (eta[test] <= tau)).sum())
            n_sil = int(silent[test].sum())
            slot = out[a]
            slot["missed"] += miss
            slot["silent"] += n_sil
            slot["flag_all"] += int(np.isinf(tau))
            if n_sil:
                slot["split_rates"].append(miss / n_sil)
    return {f"alpha={a}": {"miss_pooled": s["missed"] / s["silent"] if s["silent"] else None,
                           "miss_split_sd": float(np.std(s["split_rates"])) if s["split_rates"] else None,
                           "flag_all_share": s["flag_all"] / N_SPLITS, "n_silent_test": s["silent"]}
            for a, s in out.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pickle", type=Path, default=PICKLE)
    args = parser.parse_args()
    if not args.pickle.exists():
        raise FileNotFoundError(f"{args.pickle} not found; run warmstart_fallback.py --bench b3 first")
    with open(args.pickle, "rb") as fh:
        raw = pickle.load(fh)
    rng = np.random.default_rng(SEED)
    result, n_cal = {}, len(raw["sets"]["val"]["rows"][next(iter(raw["sets"]["val"]["rows"]))])
    for key in raw["sets"]["test"]["rows"]:
        eta, silent = query_arrays(raw["sets"]["val"]["rows"][key] + raw["sets"]["test"]["rows"][key])
        result[f"{key[0]}|{key[1]}"] = {"silent_rate": float(silent.mean()), **resplit_model(eta, silent, n_cal, rng)}
    OUT.write_text(json.dumps({"n_splits": N_SPLITS, "n_cal": n_cal, "models": result}, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT)
    f = lambda x: "  --" if x is None else f"{x:.3f}"
    print(f"{'basis':<5}{'r':>3}{'p':>7}" + "".join(f"  a={a:<4}miss  sd  flagall" for a in ALPHAS))
    for name, r in result.items():
        kind, rr = name.split("|")
        row = f"{kind:<5}{rr:>3}{r['silent_rate']:>7.3f}"
        for a in ALPHAS:
            c = r[f"alpha={a}"]
            row += f"  {f(c['miss_pooled'])} {f(c['miss_split_sd'])} {c['flag_all_share']:5.2f}   "
        print(row)


if __name__ == "__main__":
    main()
