"""Generate graded-shift query pools (experiment Q): queries a given distance outside the training box, with truth.

For each shift size ``s`` in ``config.SHIFTS`` (fractions of the box width) a pool of ``config.N_SHIFT[problem]`` queries:
``s = 0``: uniform in the training box (in-distribution baseline, fresh queries, independent of the audit pools);
``s > 0``: one parameter, chosen uniformly, is placed exactly ``s`` box widths beyond the lower or the upper end of its
range (side chosen uniformly), the others uniform in the box, so the l_inf distance from the box is exactly ``s``.
The reduced models are rebuilt as in the audit (same seed; ``eta`` of the audit pool queries is not needed here, the
models are deterministic given the seed). For every query the full-order problem is solved once; per NMF model the
reduced answer gives ``eta``, the features, the error and the convergence flag.
    python scripts/generate_shift.py --problem block --seeds 0-7
Output: ``results/shift/<problem>_seed<k>.npz``. Existing files are kept unless ``--overwrite`` is given.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import build_models                                          # noqa: E402
from contactscreen.config import (N_SHIFT, PROBLEMS, SEEDS, SHIFT_DIR, SHIFTS, make_problem, make_settings,   # noqa: E402
                                  shift_path)
from contactscreen.features import query_features                                     # noqa: E402
from contactscreen.fom import FullOrderSolver, energy_error                           # noqa: E402
from contactscreen.rom import admissibility_indicator                                 # noqa: E402

log = logging.getLogger("generate_shift")


def parse_seeds(text: str) -> list:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def shifted_queries(problem, shift: float, n: int, rng) -> np.ndarray:
    """``n`` parameters exactly ``shift`` box widths outside the box in one coordinate (inside if ``shift == 0``)."""
    mus = problem.sample(rng, n)
    if shift == 0.0:
        return mus
    width = problem.hi - problem.lo
    axis = rng.integers(0, len(problem.lo), size=n)
    upper = rng.integers(0, 2, size=n).astype(bool)
    for i in range(n):
        mus[i, axis[i]] = problem.hi[axis[i]] + shift * width[axis[i]] if upper[i] else problem.lo[axis[i]] - shift * width[axis[i]]
    return mus


def distance_outside(problem, mus: np.ndarray) -> np.ndarray:
    """l_inf distance from the training box in units of the box width (0 inside)."""
    width = problem.hi - problem.lo
    return np.max(np.maximum(problem.lo - mus, mus - problem.hi) / width, axis=1).clip(min=0.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    problem = make_problem(args.problem)
    solver = FullOrderSolver(problem)
    SHIFT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in parse_seeds(args.seeds):
        path = shift_path(args.problem, seed)
        if path.exists() and not args.overwrite:
            log.info("%s exists, skipping", path.name)
            continue
        start = time.perf_counter()
        settings = make_settings(args.problem, seed)
        models = {k: m for k, m in build_models(problem, solver, problem.sample(np.random.default_rng(settings.seed),
                                                                                  settings.n_train), settings).items() if k[0] == "nmf"}
        rng = np.random.default_rng(SEEDS["shift"] + seed)
        mus = np.concatenate([shifted_queries(problem, s, N_SHIFT[args.problem], rng) for s in SHIFTS])
        shift = np.repeat(SHIFTS, N_SHIFT[args.problem])
        dist = distance_outside(problem, mus)
        if not np.allclose(dist, shift, atol=1e-9):
            raise RuntimeError("generated queries are not at the requested distance from the box")
        arrays = {"mu": mus, "shift": shift}
        per = {k: {"X": [], "eta": np.zeros(len(mus)), "err": np.zeros(len(mus)), "converged": np.zeros(len(mus))} for k in models}
        for i, mu in enumerate(mus):
            truth = solver.solve(mu)
            if not truth.converged:
                raise RuntimeError(f"full-order solve did not converge at shifted query {i}")
            for key, model in models.items():
                red = model.solve(mu, truth.K)
                d = per[key]
                d["X"].append(query_features(problem, mu, truth.K, red))
                d["eta"][i] = admissibility_indicator(problem, mu, red)[0]
                d["err"][i], d["converged"][i] = energy_error(truth.K, truth.u, red.u), float(red.converged)
            if (i + 1) % 200 == 0:
                log.info("%s seed %d: %d/%d queries", args.problem, seed, i + 1, len(mus))
        for (kind, r), d in per.items():
            for field, value in d.items():
                arrays[f"{kind}|{r}|{field}"] = np.array(value)
        np.savez_compressed(path, **arrays)
        log.info("%s done in %.0f s", path.name, time.perf_counter() - start)


if __name__ == "__main__":
    main()
