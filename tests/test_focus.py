"""Failure-focused proposal: known density ratio and validity of weighted conformal screening with it."""

import numpy as np

from contactscreen.conformal import weighted_thresholds
from contactscreen.focus import FailureModel, FocusProposal

LO, HI = np.array([0.0, 0.0]), np.array([1.0, 1.0])


def failure_probability(mus):
    return 1.0 / (1.0 + np.exp(-8.0 * (mus[:, 0] + 0.5 * mus[:, 1] - 1.0)))


def test_failure_model_recovers_a_smooth_failure_probability():
    rng = np.random.default_rng(0)
    mus = rng.random((4000, 2))
    failed = rng.random(4000) < failure_probability(mus)
    pred = FailureModel.fit(mus, failed, LO, HI).predict(mus)
    assert np.corrcoef(pred, failure_probability(mus))[0, 1] > 0.95


def test_weights_are_bounded_and_have_unit_mean_under_the_proposal():
    rng = np.random.default_rng(1)
    prop = FocusProposal(FailureModel(LO, HI, np.array([-3.0, 4.0, 3.0, 0.0, 0.0])), lam=0.3)
    mus = prop.sample(rng, 40000)
    w = prop.weights(mus)
    assert w.max() <= prop.max_weight + 1e-12 and abs(w.mean() - 1.0) < 0.03        # E_q[p/q] = 1


def test_proposal_concentrates_on_high_failure_probability():
    rng = np.random.default_rng(2)
    prop = FocusProposal(FailureModel(LO, HI, np.array([-3.0, 4.0, 3.0, 0.0, 0.0])), lam=0.3)
    assert prop.model.predict(prop.sample(rng, 5000)).mean() > prop.model.predict(LO + (HI - LO) * rng.random((5000, 2))).mean()


def test_weighted_screening_keeps_the_miss_rate_under_the_deployment_distribution():
    """Calibrate on the focused sample with per-query weights; the miss rate under uniform queries stays <= alpha."""
    rng = np.random.default_rng(3)
    prop = FocusProposal(FailureModel(LO, HI, np.array([-3.0, 4.0, 3.0, 0.0, 0.0])), lam=0.3, floor=0.05)
    alpha, n, reps = 0.10, 150, 1500
    miss = silent = 0
    for _ in range(reps):
        cal = prop.sample(rng, n)
        fail_cal = rng.random(n) < failure_probability(cal)
        score = lambda x: x[:, 0] + 0.3 * rng.standard_normal(len(x))          # score depends on the parameters
        s_cal, w_cal = score(cal)[fail_cal], prop.weights(cal)[fail_cal]
        test = LO + (HI - LO) * rng.random((200, 2))
        fail_test = rng.random(200) < failure_probability(test)
        tau = weighted_thresholds(s_cal, w_cal, prop.weights(test), alpha)
        miss += int((fail_test & (score(test) < tau)).sum())
        silent += int(fail_test.sum())
    assert miss / silent <= alpha + 0.015
