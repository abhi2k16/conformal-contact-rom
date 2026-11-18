"""Learned-score features and scorer."""

import numpy as np

from contactscreen.features import FeatureScorer, feature_names, query_features
from contactscreen.problems import Membrane
from contactscreen.rom import ReducedSolution


def test_feature_vector_has_the_named_length_and_is_finite():
    problem = Membrane(n=15)
    mu = problem.sample(np.random.default_rng(0), 1)[0]
    sol = ReducedSolution(np.zeros(problem.n_dof), np.zeros(problem.n_contact), True, 0.0)
    x = query_features(problem, mu, problem.stiffness(mu), sol)
    assert len(x) == len(feature_names(problem)) and np.all(np.isfinite(x))


def test_scorer_ranks_failures_above_good_answers_on_separable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    silent = X[:, 0] + 0.3 * rng.normal(size=300) > 0
    scorer = FeatureScorer.fit(X, silent)
    s = scorer.score(rng.normal(size=(200, 4)) + np.array([1.0, 0, 0, 0]))
    assert scorer.cv_auc > 0.9 and s.mean() > scorer.score(rng.normal(size=(200, 4)) - np.array([1.0, 0, 0, 0])).mean()


def test_scorer_refuses_to_fit_without_enough_of_both_classes():
    assert FeatureScorer.fit(np.random.default_rng(0).normal(size=(50, 3)), np.zeros(50, bool)) is None
