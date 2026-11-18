"""Condensed (Delassus) solver agrees with the suite's full-order solver; warm starts do not change the answer."""

import numpy as np
import pytest

from contactscreen.condensed import CondensedSolver, solve_lcp
from contactscreen.fom import FullOrderSolver
from contactscreen.problems import ElasticBlock, Membrane

CONSTANT_K = [Membrane(n=15), ElasticBlock(n_xy=6, nz=2)]


@pytest.mark.parametrize("problem", CONSTANT_K, ids=lambda p: p.name)
def test_condensed_solution_matches_the_suite_solver(problem):
    mu_list = problem.sample(np.random.default_rng(0), 5)
    reference = FullOrderSolver(problem)
    condensed = CondensedSolver(problem.stiffness(mu_list[0]), problem.B)
    for mu in mu_list:
        ref = reference.solve(mu)
        sol = condensed.solve(problem.load(mu), problem.offset(mu))
        scale = np.abs(ref.u).max()
        assert np.abs(sol.u - ref.u).max() < 1e-8 * scale
        assert np.abs(sol.lam - ref.lam).max() < 1e-7 * max(1.0, np.abs(ref.lam).max())
        assert sol.gap.min() > -1e-9 * scale and sol.lam.min() >= 0.0


@pytest.mark.parametrize("problem", CONSTANT_K, ids=lambda p: p.name)
def test_warm_start_gives_the_same_answer_whatever_the_seed(problem):
    mu = problem.sample(np.random.default_rng(1), 1)[0]
    condensed = CondensedSolver(problem.stiffness(mu), problem.B)
    f, c = problem.load(mu), problem.offset(mu)
    cold = condensed.solve(f, c)
    rng = np.random.default_rng(2)
    for seed in (cold.active_mask, ~cold.active_mask, rng.random(problem.n_contact) < 0.5):
        warm = condensed.solve(f, c, active0=seed)
        assert np.abs(warm.u - cold.u).max() < 1e-8 * np.abs(cold.u).max()


def test_lcp_rejects_a_wrongly_shaped_matrix():
    with pytest.raises(ValueError):
        solve_lcp(np.eye(3), np.zeros(4))


def test_lcp_solves_a_hand_example():
    """H = I: lam_i = max(-q_i, 0)."""
    q = np.array([-2.0, 3.0, -0.5, 0.0])
    lam, gap, _, _ = solve_lcp(np.eye(4), q)
    assert np.allclose(lam, [2.0, 0.0, 0.5, 0.0]) and np.allclose(gap, [0.0, 3.0, 0.0, 0.0])
