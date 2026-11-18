"""Reduced contact model (``rom_engine.ContactROM``) and the cheap online admissibility indicator.

``u = V u_r``, ``lam = U lam_r``; the reduced problem is ``A_r u_r = V^T f + B_r^T lam_r``, ``B_r u_r >= c_r``,
``lam_r >= 0`` with ``A_r = V^T K V``, ``B_r = U^T B V``, ``c_r = U^T c`` (Balajewicz et al.). The reduced answer can
satisfy these conditions exactly and still violate the full ones: ``B V u_r < c`` (penetration) or ``U lam_r < 0`` for a
signed dual basis. That is the silent failure the indicator is meant to catch.
"""

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from rom_engine import ContactROM

from .problems import ContactProblem

ACTIVE_TOL = 1e-12
TINY = 1e-300                    # division guard only
E_LAMBDA = 0.05                  # indicator v2: multiplier scale never below this fraction of ||f||


@dataclass
class ReducedSolution:
    u: np.ndarray                 # lifted displacement
    lam: np.ndarray               # lifted multiplier (may be negative for a signed dual basis)
    converged: bool
    seconds: float                # online time: reduced assembly if K varies, plus the reduced LCP solve

    @property
    def active(self) -> np.ndarray:
        return np.flatnonzero(self.lam > ACTIVE_TOL)


class ReducedModel:
    """Reduced model for one primal basis ``V`` and one dual basis ``U``."""

    def __init__(self, problem: ContactProblem, V: np.ndarray, U: np.ndarray, kind: str):
        self.problem, self.V, self.U, self.kind = problem, V, U, kind
        self.r_u, self.r_lambda = V.shape[1], U.shape[1]
        self._shared = None if problem.k_varies else ContactROM(problem.stiffness(problem.lo), problem.B, V, U)

    def solve(self, mu: np.ndarray, K: sp.spmatrix | None = None) -> ReducedSolution:
        """Reduced solve at ``mu``; ``K`` is the stiffness at ``mu`` (needed when it varies)."""
        p = self.problem
        f, c = p.load(mu), p.offset(mu)
        start = time.perf_counter()
        rom = self._shared if self._shared is not None else ContactROM(K, p.B, self.V, self.U)
        res = rom.solve(f, c)
        seconds = time.perf_counter() - start
        return ReducedSolution(res.u, res.lam, bool(res.ok), seconds)


def admissibility_indicator(problem: ContactProblem, mu: np.ndarray, sol: ReducedSolution, version: int = 1) -> tuple:
    """Dimensionless violation of the full contact conditions by a lifted reduced solution, and its cost in seconds.

    ``eta = ||lam_-|| / s_lam + ||g_-|| / s_g + sqrt(sum lam_+ |g| / (s_lam s_g))`` with the lifted multiplier ``lam``,
    the full gap ``g = B u - c``. Version 1 scales by the answer itself, ``s_lam = ||lam||``, ``s_g = ||c||``. Version 2 uses
    problem-level floors, ``s_lam = max(||lam||, E_LAMBDA ||f||)``, ``s_g = max(||c||, char_gap)``, so it is stable when
    the lifted multiplier is tiny or the floor is flat. One sparse product with ``B``, no stiffness solve. ``eta = 0``
    means the lifted pair is admissible.
    """
    start = time.perf_counter()
    c = problem.offset(mu)
    gap = problem.B @ sol.u - c
    lam = sol.lam
    if version == 1:
        s_lam, s_gap = max(np.linalg.norm(lam), TINY), max(np.linalg.norm(c), TINY)
    elif version == 2:                              # problem-level floors: stable for lam~ ~ 0 and for a flat floor c = 0
        s_lam = max(np.linalg.norm(lam), E_LAMBDA * np.linalg.norm(problem.load(mu)), TINY)
        s_gap = max(np.linalg.norm(c), problem.char_gap, TINY)
    else:
        raise ValueError("version must be 1 or 2")
    neg = np.linalg.norm(np.minimum(lam, 0.0)) / s_lam
    viol = np.linalg.norm(np.minimum(gap, 0.0)) / s_gap
    comp = np.sqrt(np.sum(np.maximum(lam, 0.0) * np.abs(gap)) / s_lam / s_gap)      # two divisions: the product of two guards underflows
    return float(neg + viol + comp), time.perf_counter() - start
