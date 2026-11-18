"""Per-query audit: truth (full-order solve), every reduced model's answer, error and indicator.

An audit is one replicate of the whole pipeline: draw training parameters, solve the full-order model for snapshots,
build the bases and reduced models, then evaluate a pool of in-range queries and (optionally) out-of-range queries.
Result: a flat dict of NumPy arrays, saved with ``save_audit`` (``.npz``) and read back with ``load_audit``.
Keys: ``<tag>|<kind>|<r_lambda>|<field>`` per model, ``<tag>|fom|<field>`` for the full-order solve, ``<tag>|mu``.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bases import dual_basis, pod_basis
from .fom import FullOrderSolver, energy_error
from .problems import ContactProblem
from .rom import ReducedModel, admissibility_indicator

log = logging.getLogger(__name__)

FIELDS = ("eta", "eta2", "err", "converged", "t_rom", "t_eta", "m_rom")           # per query and model
FOM_FIELDS = ("it", "m", "t")                                              # full-order iterations, active set, time


@dataclass
class AuditSettings:
    n_train: int = 60
    r_primal: int = 20
    models: tuple = (("svd", 12), ("nmf", 8), ("nmf", 16))        # (dual basis kind, dual dimension)
    n_pool: int = 300
    n_outside: int = 0
    seed: int = 0


@dataclass
class Audit:
    arrays: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def build_models(problem: ContactProblem, solver: FullOrderSolver, mu_train: np.ndarray, s: AuditSettings) -> dict:
    """``{(kind, r_lambda): ReducedModel}`` from full-order snapshots at ``mu_train``."""
    sols = [solver.solve(mu) for mu in mu_train]
    if not all(sol.converged for sol in sols):
        raise RuntimeError("a full-order training solve did not converge")
    V = pod_basis(np.column_stack([sol.u for sol in sols]), s.r_primal)
    lam_snap = np.column_stack([sol.lam for sol in sols])
    return {(kind, r): ReducedModel(problem, V, dual_basis(kind, lam_snap, r), kind) for kind, r in s.models}


def evaluate_pool(problem: ContactProblem, solver: FullOrderSolver, models: dict, mus: np.ndarray, tag: str) -> dict:
    """Arrays for the queries ``mus`` (see the module docstring for the keys)."""
    out = {f"{tag}|{kind}|{r}|{name}": np.zeros(len(mus)) for kind, r in models for name in FIELDS}
    out.update({f"{tag}|fom|{name}": np.zeros(len(mus)) for name in FOM_FIELDS})
    for i, mu in enumerate(mus):
        sol = solver.solve(mu)
        if not sol.converged:
            raise RuntimeError(f"full-order solve did not converge at query {i}")
        for name, value in zip(FOM_FIELDS, (sol.iterations, len(sol.active), sol.seconds)):
            out[f"{tag}|fom|{name}"][i] = value
        for (kind, r), model in models.items():
            red = model.solve(mu, sol.K)
            eta, t_eta = admissibility_indicator(problem, mu, red)
            eta2, _ = admissibility_indicator(problem, mu, red, version=2)
            row = {"eta": eta, "eta2": eta2, "err": energy_error(sol.K, sol.u, red.u), "converged": float(red.converged),
                   "t_rom": red.seconds, "t_eta": t_eta, "m_rom": len(red.active)}
            for name in FIELDS:
                out[f"{tag}|{kind}|{r}|{name}"][i] = row[name]
        if (i + 1) % 100 == 0:
            log.info("%s: %d/%d queries", tag, i + 1, len(mus))
    return out


def run_audit(problem: ContactProblem, s: AuditSettings) -> Audit:
    """One replicate: train, then evaluate the in-range pool and the out-of-range pool."""
    rng = np.random.default_rng(s.seed)
    solver = FullOrderSolver(problem)
    models = build_models(problem, solver, problem.sample(rng, s.n_train), s)
    audit = Audit(meta={"problem": problem.name, "settings": s.__dict__ | {"models": [list(m) for m in s.models]},
                        "n_dof": problem.n_dof, "n_contact": problem.n_contact})
    pool = problem.sample(rng, s.n_pool)
    audit.arrays["pool|mu"] = pool
    audit.arrays.update(evaluate_pool(problem, solver, models, pool, "pool"))
    if s.n_outside:
        outside = problem.sample_outside(rng, s.n_outside)
        audit.arrays["out|mu"] = outside
        audit.arrays.update(evaluate_pool(problem, solver, models, outside, "out"))
    return audit


def save_audit(audit: Audit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **audit.arrays, meta=json.dumps(audit.meta))


def load_audit(path: Path) -> Audit:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return Audit({k: data[k] for k in data.files if k != "meta"}, meta)
