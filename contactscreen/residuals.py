"""Residual-based quantities of a lifted reduced answer: projection features (experiment S) and a deterministic bound (T).

Stationarity residual of the lifted pair: ``r = K u~ - f - B^T lam~``. For the Galerkin reduced model ``V^T r = 0`` when
the reduced solve converged, so residual features must live in a subspace orthogonal to ``V``.

Deterministic bound (manuscript Section 3.4). With ``e = u - u~``, ``K e = B^T (lam - lam~) - r`` and
``||e||_K^2 = -g . lam~ - g~ . lam + g~ . lam~ - e^T r``. If the lifted pair is feasible (``g~ >= 0``, ``lam~ >= 0``) the
first two terms are not positive, so ``||e||_K^2 <= ||e||_K ||r||_{K^-1} + g~ . lam~`` and
``||e||_K <= ||r||_2 / sqrt(lambda_min(K)) + sqrt(g~ . lam~)``. Relative to ``||u||_K >= ||u~||_K - bound``. For an
infeasible pair the bound needs the unknown full-order multiplier and is not computable (returned as ``inf``).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .problems import ContactProblem
from .rom import ReducedSolution

FEASIBILITY_TOL = 1e-12          # relative tolerance on negative gap / multiplier entries (rounding only)


def residual(problem: ContactProblem, mu: np.ndarray, K: sp.spmatrix, sol: ReducedSolution) -> np.ndarray:
    """Stationarity residual ``K u~ - f - B^T lam~`` of the lifted reduced answer."""
    return K @ sol.u - problem.load(mu) - problem.B.T @ sol.lam


def projection_basis(R: np.ndarray, k: int) -> np.ndarray:
    """Leading ``k`` left singular vectors of the residual snapshots ``R`` (columns); orthonormal."""
    if R.ndim != 2 or R.shape[1] < 1:
        raise ValueError("R must be a 2-D array with at least one column")
    U, _, _ = np.linalg.svd(R, full_matrices=False)
    return U[:, :min(k, U.shape[1])]


def min_eigenvalue(K: sp.spmatrix) -> float:
    """Smallest eigenvalue of an SPD sparse matrix (shift-invert Lanczos around zero)."""
    value = spla.eigsh(sp.csc_matrix(K), k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0]
    if value <= 0:
        raise ValueError(f"matrix is not positive definite (lambda_min = {value})")
    return float(value)


class StiffnessFloor:
    """Lower bound of ``lambda_min(K(mu))``: exact for constant ``K``; for the layered block ``K = K_l + rho K_u``,
    ``lambda_min(K) >= min(1, rho) lambda_min(K_l + K_u)`` because ``K_l + rho K_u >= min(1, rho) (K_l + K_u)``."""

    def __init__(self, problem: ContactProblem):
        self.layered = bool(getattr(problem, "layered", False))
        mid = 0.5 * (problem.lo + problem.hi)
        if self.layered:
            mid = mid.copy()
            mid[0] = 1.0                              # rho = 1 gives K_lower + K_upper
        self.base = min_eigenvalue(problem.stiffness(mid))

    def __call__(self, mu: np.ndarray) -> float:
        return self.base * (min(1.0, float(mu[0])) if self.layered else 1.0)


@dataclass
class BoundResult:
    feasible: bool
    abs_bound: float             # bound on ||u - u~||_K (inf if not computable)
    rel_bound: float             # bound on ||u - u~||_K / ||u||_K (inf if not computable)


def residual_bound(problem: ContactProblem, mu: np.ndarray, K: sp.spmatrix, sol: ReducedSolution,
                   lam_min: float) -> BoundResult:
    """Deterministic energy-error bound of a lifted reduced answer (module docstring)."""
    c = problem.offset(mu)
    gap = problem.B @ sol.u - c
    lam = sol.lam
    tol_g = FEASIBILITY_TOL * max(np.abs(c).max(), 1e-300)
    tol_l = FEASIBILITY_TOL * max(np.abs(lam).max(), 1e-300)
    if gap.min() < -tol_g or lam.min() < -tol_l:
        return BoundResult(False, np.inf, np.inf)
    r = residual(problem, mu, K, sol)
    bound = float(np.linalg.norm(r) / np.sqrt(lam_min) + np.sqrt(max(float(np.maximum(lam, 0) @ np.maximum(gap, 0)), 0.0)))
    norm_u = float(np.sqrt(max(sol.u @ (K @ sol.u), 0.0)))
    rel = bound / (norm_u - bound) if norm_u > bound else np.inf
    return BoundResult(True, bound, rel)
