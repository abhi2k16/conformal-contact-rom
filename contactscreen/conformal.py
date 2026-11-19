"""Screening rules for silent failures and their evaluation.

A reduced answer is *silent-failing* if the reduced solver reports success but the energy error exceeds ``EPS``. The
screen flags a query for a fallback full-order solve when ``eta >= tau`` (or when the reduced solver reports failure
itself). A *miss* is a silent failure with ``eta < tau`` (strict, so ties are flagged: conservative).

Rules
  * ``conformal_threshold``  : class-conditional split conformal. With ``n`` calibration silent failures and
        ``k = floor(alpha (n + 1))``, ``tau`` is the ``k``-th smallest failure score. If the calibration failures and a
        new silent failure are exchangeable, ``P(miss | silent failure) <= k / (n + 1) <= alpha``. Needs
        ``n >= ceil(1 / alpha) - 1``; otherwise ``tau = -inf`` (flag every query).
  * ``weighted_threshold``   : the same with importance weights ``w_i`` (known density ratio target / calibration) and a
        worst-case weight ``w_test`` for the new point (weighted conformal prediction, Tibshirani et al. 2019, applied
        to the failure class); equal weights reproduce the rule above.
  * ``risk_threshold``       : conformal risk control for the marginal risk ``P(silent failure and miss) <= delta``,
        using every calibration query. Needs ``n >= 1 / delta - 1`` queries in total, not that many failures.
  * ``cross_conformal_flags``: class-conditional rule for a score fitted by K-fold cross-fitting on the calibration
        set itself (CV+ of Barber et al. 2021 applied to the failure class). Each calibration failure is scored by the
        fold model that did not see it, and compared with the test query scored by that same model. Theory gives
        ``P(miss | silent failure) <= 2 alpha`` (plus a small term); ``alpha`` is the typical behaviour, to be checked.
  * ``heuristic_threshold``  : ``margin * min`` of the calibration failure scores, flag if ``eta > tau``; no guarantee,
        undefined without a failure (then every query is flagged).
"""

import math
from dataclasses import dataclass

import numpy as np

EPS = 0.05                      # energy error above which a reduced answer counts as a failure
ALPHAS = (0.05, 0.10)


def min_failures(alpha: float) -> int:
    """Smallest number of calibration failures for which the class-conditional rule is defined."""
    return math.ceil(1.0 / alpha) - 1


def conformal_threshold(fail_scores: np.ndarray, alpha: float) -> float:
    n = len(fail_scores)
    k = int(math.floor(alpha * (n + 1)))
    return -np.inf if k < 1 else float(np.sort(fail_scores)[k - 1])


def weighted_threshold(fail_scores: np.ndarray, weights: np.ndarray, w_test: float, alpha: float) -> float:
    """Largest failure score ``t`` with ``(sum of weights of scores < t) + w_test <= alpha (sum of weights + w_test)``."""
    if len(fail_scores) == 0:
        return -np.inf
    order = np.argsort(fail_scores, kind="stable")
    s, w = fail_scores[order], weights[order]
    budget = alpha * (w.sum() + w_test) - w_test
    below = np.concatenate([[0.0], np.cumsum(w)[:-1]])          # weight strictly before each sorted entry (ties: over-counted, conservative)
    ok = np.flatnonzero(below <= budget)
    return -np.inf if ok.size == 0 else float(s[ok[-1]])


def weighted_thresholds(fail_scores: np.ndarray, weights: np.ndarray, w_tests: np.ndarray, alpha: float) -> np.ndarray:
    """``weighted_threshold`` for many test weights at once (one threshold per test query, using that query's own
    known weight, which is valid because the density ratio is a function of the query parameters)."""
    w_tests = np.asarray(w_tests, float)
    if len(fail_scores) == 0:
        return np.full(len(w_tests), -np.inf)
    order = np.argsort(fail_scores, kind="stable")
    s, w = fail_scores[order], weights[order]
    below = np.concatenate([[0.0], np.cumsum(w)[:-1]])
    budget = alpha * (w.sum() + w_tests) - w_tests
    j = np.searchsorted(below, budget, side="right") - 1          # last sorted entry whose preceding weight fits the budget
    return np.where(j >= 0, s[np.clip(j, 0, len(s) - 1)], -np.inf)


def risk_threshold(scores: np.ndarray, is_failure: np.ndarray, delta: float) -> float:
    n = len(scores)
    m_max = int(math.floor(delta * (n + 1) - 1))
    if m_max < 0:
        return -np.inf
    fail = np.sort(scores[is_failure])
    return float(fail[m_max]) if len(fail) > m_max else np.inf


def auc(score_pos: np.ndarray, score_neg: np.ndarray):
    """Probability that a random silent failure has a larger score than a random good answer (ties count one half);
    ``None`` if either class is empty. 0.5 = no discrimination, 1 = perfect ranking."""
    if len(score_pos) == 0 or len(score_neg) == 0:
        return None
    s = np.concatenate([score_pos, score_neg])
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    i = 0
    sorted_s = s[order]
    while i < len(s):                                        # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = len(score_pos), len(score_neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def effective_sample_size(weights: np.ndarray) -> float:
    """``(sum w)^2 / sum w^2``: the number of equally weighted samples the weighted sample is worth (0 if empty)."""
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0


def heuristic_threshold(fail_scores: np.ndarray, margin: float = 0.9) -> float:
    return margin * float(fail_scores.min()) if len(fail_scores) else -np.inf


@dataclass
class QueryLabels:
    """Per-query labels of one reduced model."""

    eta: np.ndarray
    silent: np.ndarray            # reduced solver converged and error > EPS
    good: np.ndarray              # reduced solver converged and error <= EPS
    converged: np.ndarray


def labels(eta: np.ndarray, err: np.ndarray, converged: np.ndarray) -> QueryLabels:
    conv = converged > 0.5
    return QueryLabels(eta, conv & (err > EPS), conv & (err <= EPS), conv)


def flagged(lab: QueryLabels, tau: float, strict_greater: bool = False) -> np.ndarray:
    """Queries sent to the fallback: reduced solver failed, or ``eta >= tau`` (``eta > tau`` for the heuristic rule)."""
    hit = lab.eta > tau if strict_greater else lab.eta >= tau
    return ~lab.converged | hit


def outcome(lab: QueryLabels, tau: float, strict_greater: bool = False) -> dict:
    """Counts of one screened query set: misses, silent failures, false alarms, good answers, flagged queries."""
    fl = flagged(lab, tau, strict_greater)
    return {"miss": int((lab.silent & ~fl).sum()), "silent": int(lab.silent.sum()),
            "false_alarm": int((lab.good & fl).sum()), "good": int(lab.good.sum()),
            "flagged": int(fl.sum()), "n": int(len(fl)), "tau_finite": bool(np.isfinite(tau))}


def cross_conformal_flags(oof_fail_scores: np.ndarray, fail_folds: np.ndarray, test_scores: np.ndarray, alpha: float) -> tuple:
    """Suspect flags of test queries under the cross-fitted class-conditional rule.

    ``oof_fail_scores[i]``: out-of-fold score of calibration failure ``i`` (fold ``fail_folds[i]``); ``test_scores[k, j]``:
    score of test query ``j`` by the model of fold ``k``. Query ``j`` is flagged if at least ``floor(alpha (m + 1))`` of
    the ``m`` failures satisfy ``oof_i <= test_scores[fold_i, j]``; with one fold this is ``score >= conformal_threshold``.
    Returns ``(flags, defined)``; undefined (too few failures) flags every query.
    """
    m, n_test = len(oof_fail_scores), test_scores.shape[1]
    k = int(math.floor(alpha * (m + 1)))
    if k < 1:
        return np.ones(n_test, bool), False
    count = (oof_fail_scores[:, None] <= test_scores[fail_folds]).sum(axis=0)
    return count >= k, True
