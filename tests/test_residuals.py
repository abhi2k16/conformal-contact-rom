"""Residual features and the deterministic bound: orthogonality, basis, eigenvalue floor, and that the bound holds."""

import numpy as np
import pytest

from contactscreen.audit import AuditSettings, build_models
from contactscreen.fom import FullOrderSolver, energy_norm
from contactscreen.problems import ElasticBlock, Membrane
from contactscreen.residuals import (StiffnessFloor, min_eigenvalue, projection_basis, residual, residual_bound)
from contactscreen.rom import ReducedSolution


@pytest.fixture(scope="module")
def membrane_setup():
    problem = Membrane(n=15)
    solver = FullOrderSolver(problem)
    settings = AuditSettings(n_train=30, r_primal=8, models=(("nmf", 4), ("nmf", 8)), seed=0)
    models = build_models(problem, solver, problem.sample(np.random.default_rng(0), settings.n_train), settings)
    return problem, solver, models


def test_galerkin_residual_is_orthogonal_to_the_displacement_basis(membrane_setup):
    problem, _, models = membrane_setup
    model = next(iter(models.values()))
    for mu in problem.sample(np.random.default_rng(1), 5):
        K = problem.stiffness(mu)
        sol = model.solve(mu, K)
        if sol.converged:
            r = residual(problem, mu, K, sol)
            assert np.abs(model.V.T @ r).max() < 1e-8 * max(1.0, np.abs(r).max(), np.abs(problem.load(mu)).max())


def test_projection_basis_is_orthonormal():
    R = np.random.default_rng(0).normal(size=(40, 12))
    Phi = projection_basis(R, 5)
    assert Phi.shape == (40, 5) and np.allclose(Phi.T @ Phi, np.eye(5))


def test_bound_holds_for_feasible_pairs(membrane_setup):
    """The derivation, checked: for feasible lifted pairs (exact solution moved away from the obstacle, with exact and
    perturbed non-negative multipliers), the bound is at least the true energy error."""
    problem, solver, _ = membrane_setup
    lam_min = min_eigenvalue(problem.stiffness(problem.lo))
    rng = np.random.default_rng(2)
    for mu in problem.sample(rng, 6):
        K, ref = problem.stiffness(mu), solver.solve(mu)
        for scale in (1e-3, 1e-2, 1e-1):
            u_t = ref.u + scale * np.abs(ref.u).max() * rng.random(problem.n_dof)       # gap only grows: feasible
            lam_t = ref.lam * (1 + rng.random(problem.n_contact))                        # non-negative
            sol = ReducedSolution(u_t, lam_t, True, 0.0)
            b = residual_bound(problem, mu, K, sol, lam_min)
            assert b.feasible
            assert energy_norm(K, ref.u - u_t) <= b.abs_bound * (1 + 1e-10)


def test_bound_is_not_computable_for_penetrating_answers(membrane_setup):
    problem, solver, _ = membrane_setup
    mu = problem.sample(np.random.default_rng(4), 1)[0]
    ref = solver.solve(mu)
    u_t = ref.u.copy()
    u_t[problem.B.indices[0]] = problem.offset(mu)[0] - 1.0                              # push one node into the obstacle
    b = residual_bound(problem, mu, problem.stiffness(mu), ReducedSolution(u_t, ref.lam, True, 0.0), 1.0)
    assert not b.feasible and np.isinf(b.rel_bound)


def test_layered_stiffness_floor_is_below_the_true_smallest_eigenvalue():
    problem = ElasticBlock(n_xy=4, nz=2, layered=True)
    floor = StiffnessFloor(problem)
    for mu in problem.sample(np.random.default_rng(3), 4):
        assert floor(mu) <= min_eigenvalue(problem.stiffness(mu)) * (1 + 1e-10)
