"""Full-order contact solve through ``fea_engine.solve_contact_static_sparse`` (primal-dual active set).

The suite solver works on the sparse stiffness directly (route ``reduced`` for node-selection constraints), supports a
warm start ``active0`` and reports the iteration count; this module only wraps it and adds KKT and error measures.
"""

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from fea_engine import solve_contact_static_sparse

from .problems import ContactProblem


@dataclass
class FomSolution:
    u: np.ndarray
    lam: np.ndarray
    gap: np.ndarray                   # B u - c
    iterations: int
    converged: bool
    K: sp.csr_matrix
    f: np.ndarray
    c: np.ndarray
    active_mask: np.ndarray
    seconds: float
    route: str | None = None
    fallbacks: int = 0                # routes that failed before the one that produced the answer

    @property
    def active(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask)


class FullOrderSolver:
    """Solve ``problem`` at parameters ``mu``; ``active0`` warm-starts the active-set iteration."""

    def __init__(self, problem: ContactProblem, route: str = "auto"):
        self.problem, self.route = problem, route

    def solve(self, mu: np.ndarray, active0=None, fallback: bool = True) -> FomSolution:
        p = self.problem
        K, f, c = p.stiffness(mu), p.load(mu), p.offset(mu)
        start = time.perf_counter()
        res = solve_contact_static_sparse(K, f, p.B, c, route=self.route, active0=active0, fallback=fallback)
        seconds = time.perf_counter() - start
        mask = np.asarray(res.active_mask, bool) if res.active_mask is not None else res.lam > 0
        return FomSolution(res.u, res.lam, res.gap, int(res.iterations), bool(res.converged), K, f, c, mask, seconds,
                           res.route, len(res.fallbacks or []))


def kkt_residuals(problem: ContactProblem, sol: FomSolution) -> dict:
    """The four KKT residuals of a full-order solution (all should be ~ 0)."""
    stationarity = sol.K @ sol.u - sol.f - problem.B.T @ sol.lam
    return {"stationarity": float(np.linalg.norm(stationarity, np.inf)),
            "gap_violation": float(max(0.0, -sol.gap.min())),
            "multiplier_sign": float(max(0.0, -sol.lam.min())),
            "complementarity": float(np.max(np.abs(sol.lam * sol.gap)))}


def energy_norm(K: sp.spmatrix, v: np.ndarray) -> float:
    return float(np.sqrt(max(v @ (K @ v), 0.0)))


def energy_error(K: sp.spmatrix, u_exact: np.ndarray, u_approx: np.ndarray) -> float:
    """Relative energy-norm error ``||u - u~||_K / ||u||_K`` (1 if only the exact solution is zero)."""
    denom = energy_norm(K, u_exact)
    return energy_norm(K, u_exact - u_approx) / denom if denom > 0 else float(energy_norm(K, u_approx) > 0)


def time_call(func, repeats: int = 1) -> tuple:
    """``(best wall time in s, last result)`` over ``repeats`` calls."""
    best, out = np.inf, None
    for _ in range(repeats):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return float(best), out
