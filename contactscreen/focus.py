"""Failure-focused calibration sampling with an exactly known density ratio (experiment F).

Calibration queries are drawn from ``q = lam p + (1 - lam) p h / Z``: a mixture of the deployment distribution ``p``
(uniform in the training box) and ``p`` tilted towards a predicted failure region, ``h(mu)`` = predicted probability of
a silent failure (floored), ``Z = E_p[h]``. The ratio ``w = p / q = 1 / (lam + (1 - lam) h / Z) <= 1 / lam`` is known,
so weighted conformal prediction applies (see ``conformal.weighted_thresholds``). ``h`` must be fitted on data that
are independent of the calibration and test queries.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _features(mus: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    z = (mus - lo) / (hi - lo)
    return np.column_stack([np.ones(len(z)), z, z ** 2])


@dataclass
class FailureModel:
    """Ridge logistic regression of the silent-failure indicator on quadratic features of the parameters."""

    lo: np.ndarray
    hi: np.ndarray
    coef: np.ndarray

    @classmethod
    def fit(cls, mus: np.ndarray, failed: np.ndarray, lo: np.ndarray, hi: np.ndarray, ridge: float = 1e-2) -> "FailureModel":
        X, y = _features(mus, lo, hi), failed.astype(float)

        def nll(w):
            z = X @ w
            return np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * ridge * np.sum(w[1:] ** 2)

        def grad(w):
            g = X.T @ (1.0 / (1.0 + np.exp(-(X @ w))) - y)
            g[1:] += ridge * w[1:]
            return g

        res = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B")
        return cls(lo, hi, res.x)

    def predict(self, mus: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(_features(np.atleast_2d(mus), self.lo, self.hi) @ self.coef)))


class FocusProposal:
    """Mixture proposal ``q = lam p + (1 - lam) p h / Z`` and its weights ``w = p / q``."""

    def __init__(self, model: FailureModel, lam: float = 0.3, floor: float = 0.02, n_mc: int = 100_000, seed: int = 0):
        if not 0.0 < lam <= 1.0:
            raise ValueError("lam must be in (0, 1]")
        self.model, self.lam, self.floor = model, lam, floor
        mc = model.lo + (model.hi - model.lo) * np.random.default_rng(seed).random((n_mc, len(model.lo)))
        self.Z = float(self._h(mc).mean())

    def _h(self, mus: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(mus), self.floor, 1.0)

    def weights(self, mus: np.ndarray) -> np.ndarray:
        return 1.0 / (self.lam + (1.0 - self.lam) * self._h(mus) / self.Z)

    @property
    def max_weight(self) -> float:
        return 1.0 / self.lam

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` parameters from ``q``: uniform with probability ``lam``, else uniform accepted with probability ``h``."""
        lo, hi = self.model.lo, self.model.hi
        out = np.empty((n, len(lo)))
        from_p = rng.random(n) < self.lam
        out[from_p] = lo + (hi - lo) * rng.random((int(from_p.sum()), len(lo)))
        need = int((~from_p).sum())
        got = []
        while sum(len(g) for g in got) < need:
            cand = lo + (hi - lo) * rng.random((max(4 * need, 64), len(lo)))
            got.append(cand[rng.random(len(cand)) < self._h(cand)])
        out[~from_p] = np.vstack(got)[:need]
        return out
