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


class _FlatFloorProblem:
    """Stub with a flat floor (c = 0) and identity constraints, to test the indicator scales."""

    import scipy.sparse as _sp
    B = _sp.identity(4, format="csr")
    char_gap = 0.5

    def offset(self, mu):
        return np.zeros(4)

    def load(self, mu):
        return np.array([1.0, 0.0, 0.0, 0.0])


def test_indicator_scales_are_stable_without_contact_and_for_a_flat_floor():
    stub = _FlatFloorProblem()
    nothing = ReducedSolution(np.ones(4), np.zeros(4), True, 0.0)               # no contact, floor 0, free of penetration
    for version in (1, 2):
        value = admissibility_indicator(stub, None, nothing, version)[0]
        assert np.isfinite(value) and value == 0.0
    penetrating = ReducedSolution(-np.ones(4), np.zeros(4), True, 0.0)           # B u - c = -1 < 0
    assert np.isfinite(admissibility_indicator(stub, None, penetrating, 1)[0])
    assert admissibility_indicator(stub, None, penetrating, 2)[0] == 4.0         # ||g_-|| / s_g = 2 / 0.5 with s_g = max(||c|| = 0, char_gap = 0.5)


def test_indicator_version_two_vanishes_for_an_exact_pair_and_rejects_unknown_versions():
    problem = Membrane(n=15)
    solver, rng = FullOrderSolver(problem), np.random.default_rng(0)
    mu = problem.sample(rng, 1)[0]
    sol = solver.solve(mu)
    exact = ReducedSolution(sol.u, sol.lam, True, 0.0)
    assert admissibility_indicator(problem, mu, exact, version=2)[0] < 1e-6
    try:
        admissibility_indicator(problem, mu, exact, version=3)
    except ValueError:
        return
    raise AssertionError("unknown version accepted")
