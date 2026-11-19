"""Screening rules: finite-sample guarantees checked by simulation."""

import numpy as np

from contactscreen.conformal import (conformal_threshold, heuristic_threshold, min_failures, risk_threshold,
                                     weighted_threshold)


def test_class_conditional_miss_probability_is_k_over_n_plus_1():
    rng = np.random.default_rng(0)
    n, alpha, reps = 40, 0.10, 40000
    k = int(np.floor(alpha * (n + 1)))
    miss = 0
    for _ in range(reps):
        cal, new = rng.exponential(size=n), rng.exponential()
        miss += new < conformal_threshold(cal, alpha)
    assert abs(miss / reps - k / (n + 1)) < 0.006
    assert k / (n + 1) <= alpha


def test_rule_undefined_below_minimum_failures():
    for alpha in (0.05, 0.10):
        n = min_failures(alpha)
        assert np.isfinite(conformal_threshold(np.arange(n, dtype=float), alpha))
        assert conformal_threshold(np.arange(n - 1, dtype=float), alpha) == -np.inf


def test_weighted_rule_with_equal_weights_matches_unweighted():
    rng = np.random.default_rng(1)
    for n in (19, 30, 77):
        s = rng.normal(size=n)
        for alpha in (0.05, 0.1):
            assert weighted_threshold(s, np.ones(n), 1.0, alpha) == conformal_threshold(s, alpha)


def test_weighted_rule_is_valid_under_known_covariate_shift():
    """Calibrate on x ~ q = 0.5 U + 0.5 (2x) (density 0.5 + x), weight w = 1/(0.5 + x) <= 2; test on U; score = x + noise."""
    rng = np.random.default_rng(2)
    n, alpha, reps = 60, 0.10, 6000
    miss = 0
    for _ in range(reps):
        x = np.where(rng.random(n) < 0.5, rng.random(n), np.sqrt(rng.random(n)))
        s = x + 0.2 * rng.standard_normal(n)
        tau = weighted_threshold(s, 1.0 / (0.5 + x), 2.0, alpha)
        miss += (rng.random() + 0.2 * rng.standard_normal()) < tau
    assert miss / reps <= alpha + 0.012


def test_risk_control_bounds_marginal_risk():
    rng = np.random.default_rng(3)
    n, delta, reps = 120, 0.02, 20000
    loss = 0
    for _ in range(reps):
        fail = rng.random(n + 1) < 0.15
        score = np.where(fail, rng.normal(1.0, 1.0, n + 1), rng.normal(0.0, 1.0, n + 1))
        tau = risk_threshold(score[:n], fail[:n], delta)
        loss += fail[n] and score[n] < tau
    assert loss / reps <= delta + 0.004
    assert risk_threshold(np.zeros(10), np.ones(10, bool), 0.05) == -np.inf        # too few queries: flag everything


def test_heuristic_threshold():
    assert heuristic_threshold(np.array([2.0, 3.0])) == 0.9 * 2.0
    assert heuristic_threshold(np.array([])) == -np.inf


def test_vectorised_weighted_thresholds_match_scalar_version():
    from contactscreen.conformal import weighted_thresholds
    rng = np.random.default_rng(4)
    s, w = rng.normal(size=50), rng.uniform(0.5, 2.0, size=50)
    w_tests = rng.uniform(0.5, 2.0, size=20)
    for alpha in (0.05, 0.1, 0.3):
        expected = [weighted_threshold(s, w, wt, alpha) for wt in w_tests]
        assert np.array_equal(weighted_thresholds(s, w, w_tests, alpha), expected)


def test_auc_known_values_and_ties():
    from contactscreen.conformal import auc
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5          # all tied
    assert auc(np.array([2.0, 5.0]), np.array([1.0, 3.0])) == 0.75         # pairs: 2>1 yes, 2>3 no, 5>1 yes, 5>3 yes
    assert auc(np.array([]), np.array([1.0])) is None
    rng = np.random.default_rng(0)
    assert abs(auc(rng.normal(size=4000), rng.normal(size=4000)) - 0.5) < 0.03


def test_effective_sample_size():
    from contactscreen.conformal import effective_sample_size
    assert effective_sample_size(np.ones(10)) == 10.0
    assert effective_sample_size(np.array([1.0, 0.0, 0.0])) == 1.0
    assert effective_sample_size(np.array([])) == 0.0
    assert effective_sample_size(np.array([1.0, 2.0, 3.0])) < 3.0


def test_cross_conformal_with_identical_fold_models_reproduces_the_split_rule():
    from contactscreen.conformal import conformal_threshold, cross_conformal_flags
    rng = np.random.default_rng(3)
    fail, test = rng.normal(size=40), rng.normal(size=200)
    folds = rng.integers(0, 5, size=40)
    for alpha in (0.05, 0.10):
        flags, defined = cross_conformal_flags(fail, folds, np.tile(test, (5, 1)), alpha)
        assert defined and np.array_equal(flags, test >= conformal_threshold(fail, alpha))
    flags, defined = cross_conformal_flags(fail[:5], folds[:5], np.tile(test, (5, 1)), 0.05)
    assert not defined and flags.all()
