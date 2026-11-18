"""Run metadata and solver bookkeeping."""

import json

import numpy as np

from contactscreen.environment import environment_info
from contactscreen.fom import FullOrderSolver
from contactscreen.problems import Membrane


def test_environment_info_is_json_serialisable_and_complete():
    info = environment_info()
    json.dumps(info)
    for key in ("python", "numpy", "scipy", "fea_engine", "rom_engine", "threads", "sparse_direct_solver", "cpu_count"):
        assert key in info


def test_full_order_solution_reports_route_and_fallbacks():
    problem = Membrane(n=15)
    sol = FullOrderSolver(problem).solve(problem.sample(np.random.default_rng(0), 1)[0])
    assert sol.route in ("reduced", "saddle", "condensed") and sol.fallbacks == 0


def test_characteristic_gap_scale_is_positive():
    assert Membrane(n=15).char_gap > 0
