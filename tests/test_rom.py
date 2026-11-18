"""Reduced model (rom_engine.ContactROM) and indicator behaviour."""

import numpy as np

from contactscreen.bases import dual_basis, pod_basis
from contactscreen.fom import FullOrderSolver, energy_error
from contactscreen.problems import ElasticBlock, Membrane
from contactscreen.rom import ReducedModel, ReducedSolution, admissibility_indicator


def test_reduced_model_recovers_the_exact_solution_with_an_ideal_basis():
    """With V = span(u*) and U = span(lam*) the exact pair solves the one-dimensional reduced problem."""
    for problem in (Membrane(n=15), ElasticBlock(n_xy=6, nz=2, layered=True)):
        solver, rng = FullOrderSolver(problem), np.random.default_rng(0)
        mu = problem.sample(rng, 1)[0]
        sol = solver.solve(mu)
        assert len(sol.active) > 0
        V = (sol.u / np.linalg.norm(sol.u))[:, None]
        U = (sol.lam / np.linalg.norm(sol.lam))[:, None]
        red = ReducedModel(problem, V, U, "ideal").solve(mu, sol.K)
        assert red.converged and energy_error(sol.K, sol.u, red.u) < 1e-8
        assert admissibility_indicator(problem, mu, red)[0] < 1e-6      # sqrt of round-off in the complementarity term


def test_indicator_positive_for_inadmissible_pair():
    problem = Membrane(n=15)
    solver, rng = FullOrderSolver(problem), np.random.default_rng(1)
    mu = problem.sample(rng, 1)[0]
    sol = solver.solve(mu)
    bad = ReducedSolution(sol.u - 1.0, sol.lam - 0.5 * sol.lam.max(), True, 0.0)      # penetrates, negative multiplier
    assert admissibility_indicator(problem, mu, bad)[0] > 0.1


def test_dual_bases_from_the_suite():
    rng = np.random.default_rng(2)
    M = np.maximum(rng.standard_normal((50, 30)), 0) * (rng.random((50, 30)) < 0.4)
    nmf = dual_basis("nmf", M, 6)
    assert nmf.min() >= 0 and np.allclose(np.linalg.norm(nmf, axis=0), 1.0)
    assert dual_basis("svd", M, 6).min() < 0
    V = pod_basis(rng.standard_normal((30, 12)), 5)
    assert np.allclose(V.T @ V, np.eye(5))
