"""Generate learned-score data: features of every query for every reduced model, for one or more audit replicates.

For a replicate the reduced models are rebuilt exactly as in the audit (same seed), then features are computed for
  * ``pool`` and ``out``: the audit's in-range and out-of-range queries (errors and convergence flags are taken from
    the audit; the recomputed standard indicator is checked against the audit value),
  * ``fit``: fresh in-range queries (seed ``SEEDS['features_fit'] + replicate``) with full-order solves, used only to
    fit the score; they are independent of the pool and out queries.
    python scripts/generate_features.py --problem membrane --seeds 0-7
Output: ``results/features/<problem>_seed<k>.npz`` (keys ``<tag>|mu`` and ``<tag>|<kind>|<r>|<field>`` with fields
``X`` (features), ``eta``, ``err``, ``converged``). Existing files are kept unless ``--overwrite`` is given.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import build_models, load_audit                              # noqa: E402
from contactscreen.config import (FEATURE_DIR, FIT_POOL, PROBLEMS, SEEDS, audit_path, features_path,   # noqa: E402
                                  make_problem, make_settings)
from contactscreen.features import query_features                                     # noqa: E402
from contactscreen.fom import FullOrderSolver, energy_error                           # noqa: E402
from contactscreen.rom import admissibility_indicator                                 # noqa: E402

log = logging.getLogger("generate_features")
ETA_CHECK_TOL = 1e-8


def parse_seeds(text: str) -> list:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def tag_arrays(problem, solver, models, mus, tag, audit=None) -> dict:
    """Features, standard indicator, error and convergence of every model on ``mus``; truth from ``audit`` if given."""
    out = {f"{tag}|mu": mus}
    for (kind, r), model in models.items():
        pre = f"{tag}|{kind}|{r}|"
        X, eta, err, conv = [], np.zeros(len(mus)), np.zeros(len(mus)), np.zeros(len(mus))
        for i, mu in enumerate(mus):
            K = problem.stiffness(mu)
            red = model.solve(mu, K)
            X.append(query_features(problem, mu, K, red))
            eta[i] = admissibility_indicator(problem, mu, red)[0]
            conv[i] = float(red.converged)
            if audit is None:
                truth = solver.solve(mu)
                err[i] = energy_error(truth.K, truth.u, red.u)
        if audit is not None:
            err, audit_conv = audit.arrays[pre + "err"], audit.arrays[pre + "converged"]
            if not np.allclose(eta, audit.arrays[pre + "eta"], rtol=ETA_CHECK_TOL, atol=1e-12) or not np.array_equal(conv, audit_conv):
                raise RuntimeError(f"{pre}: recomputed indicator differs from the audit; the models were not rebuilt identically")
        out.update({pre + "X": np.array(X), pre + "eta": eta, pre + "err": err, pre + "converged": conv})
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    problem, solver = make_problem(args.problem), None
    solver = FullOrderSolver(problem)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in parse_seeds(args.seeds):
        path = features_path(args.problem, seed)
        if path.exists() and not args.overwrite:
            log.info("%s exists, skipping", path.name)
            continue
        audit_file = audit_path(args.problem, seed)
        if not audit_file.exists():
            raise FileNotFoundError(f"{audit_file} missing; run scripts/generate_audits.py first")
        start = time.perf_counter()
        audit, settings = load_audit(audit_file), make_settings(args.problem, seed)
        models = build_models(problem, solver, problem.sample(np.random.default_rng(settings.seed), settings.n_train), settings)
        arrays = {}
        for tag in ("pool", "out"):
            arrays.update(tag_arrays(problem, solver, models, audit.arrays[f"{tag}|mu"], tag, audit))
        fit_mus = problem.sample(np.random.default_rng(SEEDS["features_fit"] + seed), FIT_POOL[args.problem])
        arrays.update(tag_arrays(problem, solver, models, fit_mus, "fit"))
        np.savez_compressed(path, **arrays)
        log.info("%s done in %.0f s", path.name, time.perf_counter() - start)


if __name__ == "__main__":
    main()
