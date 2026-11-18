"""Reduced bases from snapshots, using ``rom_engine`` for the contact-specific dual bases.

Primal: proper orthogonal decomposition (SVD) of the displacement snapshots.
Dual (contact forces):
  * ``svd`` : leading singular vectors of the multiplier snapshots (``svd_dual_basis``); signed, so a nonnegative
              reduced multiplier can lift to a negative one.
  * ``nmf`` : nonnegative matrix factorisation (``nonneg_dual_basis``, NNDSVD start); the lifted multiplier stays
              nonnegative for every nonnegative reduced multiplier.
"""

import numpy as np

from rom_engine import nonneg_dual_basis, svd_dual_basis
from rom_engine.contact_dual_basis import active_snapshot_columns

DUAL_KINDS = ("svd", "nmf")


def pod_basis(snapshots: np.ndarray, r: int) -> np.ndarray:
    """First ``r`` left singular vectors of the snapshot matrix (columns are snapshots)."""
    u, s, _ = np.linalg.svd(snapshots, full_matrices=False)
    return u[:, :min(r, int(np.sum(s > 1e-12 * s[0])))]


def dual_basis(kind: str, multiplier_snapshots: np.ndarray, r: int) -> np.ndarray:
    """Dual basis with at most ``r`` columns from the snapshots that have contact."""
    cols = active_snapshot_columns(multiplier_snapshots)
    if len(cols) == 0:
        raise ValueError("no snapshot with contact: cannot build a dual basis")
    M = multiplier_snapshots[:, cols]
    if kind == "svd":
        return svd_dual_basis(M, r)
    if kind == "nmf":
        return nonneg_dual_basis(M, r).U
    raise ValueError(f"unknown dual basis kind {kind!r}; choose from {DUAL_KINDS}")
