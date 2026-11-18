"""Condensed (Delassus) full-order contact solver for problems with a constant stiffness matrix.

With ``K`` constant and SPD, the contact problem  K u = f + B^T lam,  g = B u - c >= 0,  lam >= 0,  lam . g = 0
reduces to a linear complementarity problem of the size of the contact set only:
    g = q + H lam,   q = B K^{-1} f - c,   H = B K^{-1} B^T  (SPD, n_c x n_c),
followed by the lift  u = K^{-1} f + G^T lam  with  G = B K^{-1}.
``K`` is factorised once and ``H``, ``G`` are built once (the set-up cost); each query then costs one back-substitution
(``K^{-1} f``), a dense LCP solve and one matrix-vector product. Used to test whether a much cheaper full-order solve
removes the room for screening (plan, Section 11). The LCP is solved by block principal pivoting (Judice and Pires,
1994) with Murty's single-pivot backup rule, which terminates for the P-matrix ``H``.
"""

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MAX_BLOCK_TRIES = 10            # full block exchanges without progress before falling back to single pivots
MAX_ITERATIONS = 10_000


@dataclass
class CondensedSolution:
    u: np.ndarray
    lam: np.ndarray
    gap: np.ndarray
    iterations: int                 # linear solves with a principal block of H
    active_mask: np.ndarray
    seconds: float
    seconds_back_substitution: float
    seconds_lcp: float


class CondensedSolver:
    """Constant-``K`` contact solver; build once, call ``solve`` per query."""

    def __init__(self, K: sp.spmatrix, B: sp.spmatrix):
        start = time.perf_counter()
        self.B = sp.csr_matrix(B)
        self.lu = spla.splu(sp.csc_matrix(K))
        self.G = np.ascontiguousarray(self.lu.solve(self.B.T.toarray()).T)          # B K^{-1}, n_c x N
        self.H = np.asarray(self.B @ self.G.T)
        self.H = 0.5 * (self.H + self.H.T)                                        # remove round-off asymmetry
        self.setup_seconds = time.perf_counter() - start

    def solve(self, f: np.ndarray, c: np.ndarray, active0: np.ndarray | None = None) -> CondensedSolution:
        start = time.perf_counter()
        w = self.lu.solve(f)
        q = self.B @ w - c
        mid = time.perf_counter()
        lam, gap, iterations, active = solve_lcp(self.H, q, active0)
        stop = time.perf_counter()
        u = w + self.G.T @ lam
        return CondensedSolution(u, lam, gap, iterations, active, time.perf_counter() - start, mid - start, stop - mid)


def solve_lcp(H: np.ndarray, q: np.ndarray, active0: np.ndarray | None = None) -> tuple:
    """Solve ``g = q + H lam >= 0, lam >= 0, lam . g = 0`` for SPD ``H``; ``active0`` is the initial set of lam > 0.

    Returns ``(lam, g, iterations, active_mask)``.
    """
    n = len(q)
    if H.shape != (n, n):
        raise ValueError(f"H must be {n} x {n}, got {H.shape}")
    active = np.zeros(n, bool) if active0 is None else np.array(active0, bool)
    best, tries = n + 1, 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        lam = np.zeros(n)
        if active.any():
            idx = np.flatnonzero(active)
            lam[idx] = sla.solve(H[np.ix_(idx, idx)], -q[idx], assume_a="pos")
        gap = q + H @ lam
        wrong = np.where(active, lam < 0.0, gap < 0.0)
        count = int(wrong.sum())
        if count == 0:
            return lam, gap, iteration, active
        if count < best:
            best, tries, flip = count, 0, wrong
        elif tries < MAX_BLOCK_TRIES:
            tries, flip = tries + 1, wrong
        else:                                           # Murty: exchange only the largest infeasible index
            flip = np.zeros(n, bool)
            flip[np.flatnonzero(wrong)[-1]] = True
        active = active ^ flip
    raise RuntimeError(f"LCP did not converge in {MAX_ITERATIONS} iterations")
