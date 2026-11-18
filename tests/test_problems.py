"""Benchmarks assembled with fea_engine and the full-order solve through solve_contact_static_sparse."""

import numpy as np
import pytest
from scipy.optimize import nnls

from contactscreen.fom import FullOrderSolver, kkt_residuals
from contactscreen.problems import ElasticBlock, Membrane

SMALL = [Membrane(n=15), ElasticBlock(n_xy=6, nz=2), ElasticBlock(n_xy=6, nz=2, layered=True)]


@pytest.mark.parametrize("problem", SMALL, ids=lambda p: p.name)
def test_stiffness_is_symmetric_positive_definite(problem):
    K = problem.stiffness(problem.lo)
    assert abs(K - K.T).max() < 1e-12 * abs(K).max()
    assert np.linalg.eigvalsh(K.toarray()).min() > 0


@pytest.mark.parametrize("problem", SMALL, ids=lambda p: p.name)
def test_every_constraint_row_selects_one_unknown(problem):
    B = problem.B
    assert (B.getnnz(axis=1) == 1).all() and (B.data == 1.0).all()
    assert len(set(B.indices)) == problem.n_contact


@pytest.mark.parametrize("problem", SMALL, ids=lambda p: p.name)
def test_full_order_solution_is_a_kkt_point_and_matches_an_independent_solve(problem):
    """KKT residuals vanish, and the multipliers agree with nonnegative least squares on the condensed LCP."""
    solver, rng = FullOrderSolver(problem), np.random.default_rng(0)
    for mu in problem.sample(rng, 3):
        sol = solver.solve(mu)
        res = kkt_residuals(problem, sol)
        assert sol.converged and max(res.values()) < 1e-7 * max(1.0, np.abs(sol.u).max())
        K, B = sol.K.toarray(), problem.B.toarray()
        Kinv_Bt = np.linalg.solve(K, B.T)
        A = B @ Kinv_Bt                                   # gap = B u - c = A lam + (B K^-1 f - c)
        q = B @ np.linalg.solve(K, sol.f) - sol.c
        R = np.linalg.cholesky(0.5 * (A + A.T) + 1e-14 * np.eye(len(A))).T
        lam_ref = nnls(R, -np.linalg.solve(R.T, q))[0]
        assert np.allclose(sol.lam, lam_ref, atol=1e-6 * max(1.0, lam_ref.max()))


def test_load_pushes_the_block_down_and_contact_forces_push_up():
    problem = ElasticBlock(n_xy=6, nz=2)
    sol = FullOrderSolver(problem).solve(problem.hi * np.array([1.0, 0.5, 0.5]))
    assert sol.u[problem._top_dof].max() < 0 and sol.lam.min() >= -1e-12 and len(sol.active) > 0


def test_layered_stiffness_is_affine_in_the_ratio():
    p = ElasticBlock(n_xy=6, nz=2, layered=True)
    K1, K2 = p.stiffness(np.array([1.0, 0.2, 0.5, 0.5])), p.stiffness(np.array([3.0, 0.2, 0.5, 0.5]))
    assert abs(K2 - K1 - 2.0 * p._K_upper).max() < 1e-12


def test_warm_start_with_the_exact_active_set_reproduces_the_solution():
    p = Membrane(n=15)
    solver, rng = FullOrderSolver(p), np.random.default_rng(1)
    mu = p.sample(rng, 1)[0]
    cold = solver.solve(mu)
    warm = solver.solve(mu, active0=cold.active_mask)
    assert warm.converged and np.allclose(warm.u, cold.u, atol=1e-10) and warm.iterations <= cold.iterations
