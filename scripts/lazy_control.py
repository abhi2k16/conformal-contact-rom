"""Control for the lazy fallback solver (B3): ROM seed against an empty seed.

``warmstart_fallback.py`` seeds the lazy active-set solver with the lifted reduced solution. The saving could come from
the seed or merely from not forming all ``n_c`` columns of the compliance. This script re-times the lazy solver on the
same first ``n_timed`` test queries with an empty seed (no reduced model at all) and compares it with the ROM-seeded
medians stored in ``results/warmstart_b3.pkl``.

Run with the paper-1 interpreter, one process at a time (timings), from this project folder:
``..\\reduced-contact-reliability\\.venv\\Scripts\\python.exe scripts\\lazy_control.py``.
Writes ``results/lazy_control_b3.json``.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from warmstart_fallback import HERE, Config, build_b3, col, lcp_active_set, measure_lazy, time_cold_parts

PICKLE = HERE / "results" / "warmstart_b3.pkl"
OUT = HERE / "results" / "lazy_control_b3.json"

log = logging.getLogger("lazy_control")


def test_queries(cfg: Config, problem, n_test: int) -> np.ndarray:
    """Test parameters exactly as ``build_setup`` draws them (training draw first, then the test set)."""
    rng = np.random.default_rng(cfg.seed + 1)
    problem.sample_mu(rng, cfg.e5_n_train)
    return problem.sample_mu(rng, n_test)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not PICKLE.exists():
        raise FileNotFoundError(f"{PICKLE} not found; run warmstart_fallback.py --bench b3 first")
    with open(PICKLE, "rb") as fh:
        raw = pickle.load(fh)
    n_timed, n_test = raw["n_timed"], len(raw["sets"]["test"]["fom"])
    cfg = Config()
    problem = build_b3(cfg.b3)
    mu_test = test_queries(cfg, problem, n_test)[:n_timed]
    rows = []
    for i, mu in enumerate(mu_test):
        parts = time_cold_parts(problem, mu, args.repeats)
        z_ref, _, _ = lcp_active_set(parts["a_lcp"], parts["q"])
        u_ref = parts["kinv_f"] + parts["lu"].solve(parts["bs"].T @ z_ref)
        rows.append({"shared": parts["t_lu"] + parts["t_rhs"], **measure_lazy(parts, z_ref, u_ref, np.array([], int), args.repeats)})
        log.info("query %d/%d: lazy from empty seed %.2f s, %d columns", i + 1, len(mu_test), rows[-1]["t_lazy"],
                 rows[-1].get("n_cols", -1))
    good = [r for r in rows if not r["lazy_cycled"]]
    empty = {"n": len(rows), "n_cycled": len(rows) - len(good),
             "total_s_median": float(np.median([r["shared"] + r["t_lazy"] for r in good])) if good else None,
             "n_cols_median": float(np.median(col(good, "n_cols"))) if good else None,
             "it_median": float(np.median(col(good, "it_lazy"))) if good else None,
             "max_du": float(col(good, "du").max()) if good else None}
    ref = {}
    for key, d2 in raw["sets"]["test"]["d2"].items():
        ok = [r for r in d2 if not r["lazy_cycled"]]
        shared = np.median([r["t_lu"] + r["t_rhs"] for r in raw["sets"]["test"]["fom"] if "t_lu" in r])
        ref[f"{key[0]}|{key[1]}"] = {"total_s_median": float(shared + np.median(col(ok, "t_lazy"))),
                                      "n_cols_median": float(np.median(col(ok, "n_cols")))}
    OUT.write_text(json.dumps({"empty_seed": empty, "rom_seed": ref}, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT)
    print("empty seed:", empty)
    for name, r in ref.items():
        print(f"ROM seed {name:<7} total {r['total_s_median']:.2f} s, {r['n_cols_median']:.0f} columns")


if __name__ == "__main__":
    main()
