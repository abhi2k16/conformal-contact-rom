"""Experiment T: the deterministic residual bound as a screen, against the conformal rules (manuscript Section 3.4).

The deterministic screen accepts a reduced answer only if the bound of ``contactscreen.residuals`` is computable (the
lifted pair is feasible) and its relative value is at most ``EPS``; it needs no calibration. Reported per model on the
in-range pools (``generate_residuals.py``): share of answers with a computable bound, effectivity (bound / true error)
where computable, number of answers whose bound fell below the true error (must be 0), miss rate, false-alarm rate and
fallback share of the deterministic screen, and the penetration of the lifted answers (median over good answers and
over silent failures, relative to the largest obstacle offset). Replicates are the independent units.
    python scripts/run_bound.py --problem block
Writes ``results/bound_<problem>.json``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.config import PROBLEMS, RESIDUAL_DIR, RESULTS, SEEDS            # noqa: E402
from contactscreen.conformal import EPS                                            # noqa: E402
from contactscreen.environment import environment_info                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import summarise                                                 # noqa: E402

log = logging.getLogger("run_bound")


def model_key_list(data) -> list:
    return sorted({int(k.split("|")[2]) for k in data.files if k.startswith("pool|nmf|")})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    args = parser.parse_args()
    files = sorted(RESIDUAL_DIR.glob(f"{args.problem}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no residual files for {args.problem}; run scripts/generate_residuals.py first")
    boot_rng = np.random.default_rng(SEEDS["bootstrap"])
    models = model_key_list(np.load(files[0]))
    out = {"problem": args.problem, "n_replicates": len(files), "eps": EPS, "seeds": SEEDS,
           "environment": environment_info(), "models": {}}
    for r in models:
        rows, eff, pen_good, pen_silent, violations, feasible, computable = [], [], [], [], 0, [], []
        for path in files:
            d = np.load(path)
            g = lambda f: d[f"pool|nmf|{r}|{f}"]
            err, conv, rel = g("err"), g("converged") > 0.5, g("relbound")
            ok = np.isfinite(rel)
            silent, good = conv & (err > EPS), conv & (err <= EPS)
            fl = ~conv | ~ok | (rel > EPS)
            rows.append(np.array([(silent & ~fl).sum(), silent.sum(), (good & fl).sum(), good.sum(), fl.sum(), len(fl),
                                  1.0, 1.0], float))
            violations += int((ok & (err > rel)).sum())
            eff += list(rel[ok & (err > 0)] / err[ok & (err > 0)])
            pen_good += list(g("penetration")[good])
            pen_silent += list(g("penetration")[silent])
            feasible.append(g("feasible").mean())
            computable.append(ok.mean())
        med = lambda v: float(np.median(v)) if len(v) else None
        out["models"][f"nmf|{r}"] = {"feasible": float(np.mean(feasible)), "computable": float(np.mean(computable)),
                                    "effectivity_median": med(eff), "n_effectivity": len(eff), "violations": violations,
                                    "penetration_good_median": med(pen_good), "penetration_silent_median": med(pen_silent),
                                    "screen": summarise(rows, boot_rng)}
        log.info("model nmf %d done", r)
    path = RESULTS / f"bound_{args.problem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    for name, v in out["models"].items():
        s = v["screen"]
        print(f"{args.problem} {name}: computable {v['computable']:.3f}, effectivity {v['effectivity_median']}, violations "
              f"{v['violations']}, penetration good/silent {v['penetration_good_median']:.4f}/{v['penetration_silent_median']:.4f}; "
              f"screen miss {s['miss']}, false alarm {s['fpr']}, fallback {s['flagged']:.3f}")


if __name__ == "__main__":
    main()
