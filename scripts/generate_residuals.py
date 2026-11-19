"""Generate residual projections (experiment S) and deterministic bounds (experiment T) for the audit pools.

Per replicate the reduced models are rebuilt exactly as in the audit (same seed; the recomputed indicator ``eta`` is
checked against the audit). For every NMF model:
  * residual snapshots at the training parameters give two projection bases of ``K_MAX`` vectors: one for the full
    residual and one for its restriction to the constrained unknowns. Training parameters are independent of the
    calibration and test queries, so the bases do not touch them;
  * for every in-range pool query: the coordinates of the residual in both bases (divided by ``||f||``), and the
    deterministic bound of ``contactscreen.residuals`` (feasibility flag, absolute and relative bound), and the
    largest penetration ``max(-g~) / max|c|`` (0 for a feasible answer);
  * the largest ``|V^T r| / ||f||`` over the pool, a check of Galerkin orthogonality.
Errors and convergence flags are taken from the audit.
    python scripts/generate_residuals.py --problem block --seeds 0-7
Output: ``results/residuals/<problem>_seed<k>.npz``. Existing files are kept unless ``--overwrite`` is given.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import build_models, load_audit                              # noqa: E402
from contactscreen.config import (PROBLEMS, RESIDUAL_DIR, audit_path, make_problem, make_settings,   # noqa: E402
                                  residuals_path)
from contactscreen.fom import FullOrderSolver                                         # noqa: E402
from contactscreen.residuals import StiffnessFloor, projection_basis, residual, residual_bound   # noqa: E402
from contactscreen.rom import admissibility_indicator                                 # noqa: E402

log = logging.getLogger("generate_residuals")
K_MAX = 10
ETA_CHECK_TOL = 1e-8


def parse_seeds(text: str) -> list:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def training_bases(problem, model, mus_train) -> tuple:
    """Projection bases of the residuals of ``model`` at the training parameters: (full, contact part)."""
    R = []
    for mu in mus_train:
        K = problem.stiffness(mu)
        sol = model.solve(mu, K)
        if sol.converged:
            R.append(residual(problem, mu, K, sol) / max(np.linalg.norm(problem.load(mu)), 1e-300))
    R = np.column_stack(R)
    return projection_basis(R, K_MAX), projection_basis(R[problem.B.indices], K_MAX)


def model_arrays(problem, model, mus_pool, mus_train, floor, audit, pre) -> dict:
    phi_full, phi_contact = training_bases(problem, model, mus_train)
    n = len(mus_pool)
    out = {"cfull": np.zeros((n, phi_full.shape[1])), "ccontact": np.zeros((n, phi_contact.shape[1])),
           "feasible": np.zeros(n), "absbound": np.zeros(n), "relbound": np.zeros(n), "eta": np.zeros(n),
           "penetration": np.zeros(n)}
    worst_orth = 0.0
    for i, mu in enumerate(mus_pool):
        K = problem.stiffness(mu)
        sol = model.solve(mu, K)
        f_norm = max(np.linalg.norm(problem.load(mu)), 1e-300)
        r = residual(problem, mu, K, sol) / f_norm
        out["cfull"][i] = phi_full.T @ r
        out["ccontact"][i] = phi_contact.T @ r[problem.B.indices]
        if sol.converged:
            worst_orth = max(worst_orth, float(np.abs(model.V.T @ r).max()))
        c = problem.offset(mu)
        out["penetration"][i] = max(0.0, -float((problem.B @ sol.u - c).min())) / max(np.abs(c).max(), 1e-300)
        b = residual_bound(problem, mu, K, sol, floor(mu))
        out["feasible"][i], out["absbound"][i], out["relbound"][i] = float(b.feasible), b.abs_bound, b.rel_bound
        out["eta"][i] = admissibility_indicator(problem, mu, sol)[0]
    if not np.allclose(out["eta"], audit.arrays[pre + "eta"], rtol=ETA_CHECK_TOL, atol=1e-12):
        raise RuntimeError(f"{pre}: recomputed eta differs from the audit; models not rebuilt identically")
    out.update({"err": audit.arrays[pre + "err"], "converged": audit.arrays[pre + "converged"],
                "orthogonality": np.array(worst_orth)})
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    problem = make_problem(args.problem)
    solver, floor = FullOrderSolver(problem), StiffnessFloor(problem)
    RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    for seed in parse_seeds(args.seeds):
        path = residuals_path(args.problem, seed)
        if path.exists() and not args.overwrite:
            log.info("%s exists, skipping", path.name)
            continue
        audit_file = audit_path(args.problem, seed)
        if not audit_file.exists():
            raise FileNotFoundError(f"{audit_file} missing; run scripts/generate_audits.py first")
        start = time.perf_counter()
        audit, settings = load_audit(audit_file), make_settings(args.problem, seed)
        mus_train = problem.sample(np.random.default_rng(settings.seed), settings.n_train)
        models = build_models(problem, solver, mus_train, settings)
        arrays = {"lambda_min_base": np.array(floor.base)}
        for (kind, r), model in models.items():
            if kind != "nmf":
                continue
            pre = f"pool|{kind}|{r}|"
            for key, value in model_arrays(problem, model, audit.arrays["pool|mu"], mus_train, floor, audit, pre).items():
                arrays[pre + key] = value
        np.savez_compressed(path, **arrays)
        log.info("%s done in %.0f s", path.name, time.perf_counter() - start)


if __name__ == "__main__":
    main()
