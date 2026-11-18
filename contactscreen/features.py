"""Learned screening score (direction 1 of the plan): cheap per-query features and a ridge-logistic scorer.

The conformal rules of ``conformal.py`` are valid for *any* score computed from the query and the reduced answer, as
long as the score is fitted on data independent of the calibration and test queries. This module supplies such a
score. Features use only quantities the reduced solve already has (lifted answer, one sparse product with ``K`` and
``B``) plus the normalised parameters; no full-order solve is involved online.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from .problems import ContactProblem
from .rom import ReducedSolution, TINY

LOG_FLOOR = 1e-12
SIGN_TOL = 1e-9                 # relative threshold for "negative" / "positive" lifted multiplier entries
RIDGES = (0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5


def feature_names(problem: ContactProblem) -> list:
    d = len(problem.lo)
    return (["log_neg_lam", "log_gap_violation", "log_complementarity", "log_residual", "log_residual_contact",
             "log_lam_over_load", "share_positive_lam", "share_negative_lam"]
            + [f"z{i}" for i in range(d)] + [f"zc{i}" for i in range(d)])


def query_features(problem: ContactProblem, mu: np.ndarray, K: sp.spmatrix, sol: ReducedSolution) -> np.ndarray:
    """Feature vector of one query; the residual ``K u~ - f - B^T lam~`` is the stationarity defect of the lifted answer."""
    f, c, lam = problem.load(mu), problem.offset(mu), sol.lam
    gap = problem.B @ sol.u - c
    s_lam, s_gap, s_f = max(np.linalg.norm(lam), TINY), max(np.linalg.norm(c), TINY), max(np.linalg.norm(f), TINY)
    neg = np.linalg.norm(np.minimum(lam, 0.0)) / s_lam
    viol = np.linalg.norm(np.minimum(gap, 0.0)) / s_gap
    comp = np.sqrt(np.sum(np.maximum(lam, 0.0) * np.abs(gap)) / s_lam / s_gap)
    residual = K @ sol.u - f - problem.B.T @ lam
    contact_dofs = problem.B.indices
    log = lambda x: float(np.log10(x + LOG_FLOOR))
    z = (mu - problem.lo) / (problem.hi - problem.lo)
    return np.concatenate([[log(neg), log(viol), log(comp), log(np.linalg.norm(residual) / s_f),
                            log(np.linalg.norm(residual[contact_dofs]) / s_f), log(s_lam / s_f),
                            float(np.mean(lam > SIGN_TOL * s_lam)), float(np.mean(lam < -SIGN_TOL * s_lam))], z, (z - 0.5) ** 2])


def _expand(X: np.ndarray, degree: int) -> np.ndarray:
    if degree == 1:
        return X
    cross = [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
    return np.column_stack([X] + cross)


def _fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])

    def nll(w):
        z = A @ w
        return np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * ridge * np.sum(w[1:] ** 2)

    def grad(w):
        g = A.T @ (1.0 / (1.0 + np.exp(-(A @ w))) - y)
        g[1:] += ridge * w[1:]
        return g

    return minimize(nll, np.zeros(A.shape[1]), jac=grad, method="L-BFGS-B").x


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a positive scores above a negative (ties count one half)."""
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@dataclass
class FeatureScorer:
    """Standardise, optionally add quadratic terms, ridge-logistic fit of P(silent failure); the score is the logit.

    ``degree`` and ``ridge`` are chosen by ``N_FOLDS``-fold cross-validated AUC on the fitting data only.
    """

    mean: np.ndarray
    scale: np.ndarray
    degree: int
    ridge: float
    weights: np.ndarray
    cv_auc: float

    @classmethod
    def fit(cls, X: np.ndarray, silent: np.ndarray, seed: int = 0) -> "FeatureScorer | None":
        """``None`` if there are too few failures or good answers to fit (fewer than ``N_FOLDS`` of either)."""
        y = silent.astype(float)
        if min(y.sum(), len(y) - y.sum()) < N_FOLDS:
            return None
        mean, scale = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-12)
        Z = (X - mean) / scale
        folds = np.random.default_rng(seed).permutation(len(y)) % N_FOLDS
        best = None
        for degree in (1, 2):
            E = _expand(Z, degree)
            for ridge in RIDGES:
                aucs = []
                for k in range(N_FOLDS):
                    tr, te = folds != k, folds == k
                    if y[te].sum() == 0 or y[te].sum() == te.sum():
                        continue
                    w = _fit_logistic(E[tr], y[tr], ridge)
                    s = np.column_stack([np.ones(te.sum()), E[te]]) @ w
                    aucs.append(_auc(s[y[te] == 1], s[y[te] == 0]))
                cv = float(np.mean(aucs)) if aucs else -1.0
                if best is None or cv > best[0]:
                    best = (cv, degree, ridge)
        cv, degree, ridge = best
        return cls(mean, scale, degree, ridge, _fit_logistic(_expand(Z, degree), y, ridge), cv)

    def score(self, X: np.ndarray) -> np.ndarray:
        E = _expand((X - self.mean) / self.scale, self.degree)
        return np.column_stack([np.ones(len(E)), E]) @ self.weights
