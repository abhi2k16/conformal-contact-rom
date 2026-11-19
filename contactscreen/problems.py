"""Parametrised frictionless contact benchmarks, assembled with the suite package ``fea_engine``.

Convention of the suite (and of this project): unknowns ``u``, SPD stiffness ``K(mu)``, load ``f(mu)``, constraint rows
``B`` and offset ``c(mu)``:
    K u = f + B^T lam,   gap g = B u - c >= 0,   lam >= 0,   lam . g = 0.
Every ``B`` here selects single unknowns (rows with one positive entry), which is the structure the sparse solver
``solve_contact_static_sparse`` handles on its ``reduced`` route.

  * ``Membrane``     : Poisson membrane (``Quad4Laplace``) pushed down by a moving Gaussian load onto a curved rigid
                       floor; ``K`` constant. Parameters: peak load, load centre.
  * ``ElasticBlock`` : 3D linear-elastic block (``Hex8Solid3D``), lateral faces clamped, pressure on the top face,
                       resting on a rigid curved foundation whose centre moves. With ``layered`` the stiffness ratio of
                       an upper and a lower layer is the first parameter, so ``K(mu) = K_lower + rho K_upper`` varies.
Parameters are sampled uniformly in a training box; ``sample_outside`` draws from a region around that box.
"""

import warnings

import numpy as np
import scipy.sparse as sp

from fea_engine import FESystem, Hex8Solid3D, Quad4Laplace
from fea_engine.material import Material, D_solid3d
from fea_engine.mesh import MultiBlockMesh, box_mesh, rectangle_mesh


# Out-of-range pools (sample_outside): load scaled by a factor in OUT_LOAD_FACTOR times the training maximum, or load
# centre moved up to OUT_CENTRE_PAD beyond its range; for the layered block the stiffness ratio is also drawn from
# OUT_RATIO_FACTOR times its training maximum.
OUT_LOAD_FACTOR = (1.3, 2.0)
OUT_CENTRE_PAD = 0.15
OUT_RATIO_FACTOR = (1.2, 2.0)


class ContactProblem:
    """Interface shared by the benchmarks."""

    name: str
    k_varies: bool
    n_dof: int
    n_contact: int
    B: sp.csr_matrix
    lo: np.ndarray                      # training box of the parameters
    hi: np.ndarray

    def stiffness(self, mu: np.ndarray) -> sp.csr_matrix:
        raise NotImplementedError

    def load(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def offset(self, mu: np.ndarray) -> np.ndarray:
        """Constraint offset ``c`` (the floor height seen by the constrained unknowns)."""
        raise NotImplementedError

    @property
    def char_gap(self) -> float:
        """Characteristic gap scale: norm of the offset at the centre of the training box."""
        return float(np.linalg.norm(self.offset(0.5 * (self.lo + self.hi))))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` parameters, uniform in the training box."""
        return self.lo + (self.hi - self.lo) * rng.random((n, len(self.lo)))

    def sample_outside(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError

    def in_box(self, mus: np.ndarray) -> np.ndarray:
        """True for parameters inside the training box."""
        return np.all((mus >= self.lo) & (mus <= self.hi), axis=1)


def selection_matrix(positions: np.ndarray, n_dof: int) -> sp.csr_matrix:
    """Rows pick the unknowns at ``positions`` (one positive entry per row)."""
    m = len(positions)
    return sp.csr_matrix((np.ones(m), (np.arange(m), positions)), shape=(m, n_dof))


def sample_shell(rng, n, lo, hi, load_axis, load_factor, centre_axes, extra=None) -> np.ndarray:
    """Parameters outside the training box: load scaled up (mode 0) or load centre moved out of its range (mode 1);
    ``extra = (axis, (a, b))`` additionally draws that parameter from ``[a, b]``."""
    out = lo + (hi - lo) * rng.random((n, len(lo)))
    mode = rng.integers(0, 2, size=n)
    out[mode == 0, load_axis] = hi[load_axis] * rng.uniform(*load_factor, size=int((mode == 0).sum()))
    axes = list(centre_axes)
    for i in np.flatnonzero(mode == 1):
        while True:                                  # rejection: wider box, but outside the training box
            cand = rng.uniform(lo[axes[0]] - OUT_CENTRE_PAD, hi[axes[0]] + OUT_CENTRE_PAD, size=len(axes))
            if np.any((cand < lo[axes]) | (cand > hi[axes])):
                out[i, axes] = cand
                break
    if extra is not None:
        axis, (a, b) = extra
        out[:, axis] = rng.uniform(a, b, size=n)
    return out


# ------------------------------------------------------------------------------------------------ membrane

class Membrane(ContactProblem):
    """Membrane on an ``(n + 1) x (n + 1)`` node grid, clamped edge, ``-Laplace(u) = F`` with a downward Gaussian load.

    The floor is ``-(d0 + kappa r^2)`` (``r`` from the plate centre); every ``stride``-th interior node may touch it.
    """

    def __init__(self, n: int = 25, stride: int = 2, d0: float = 0.3, kappa: float = 1.5, sigma: float = 0.12,
                 load_range=(30.0, 200.0), centre_range=(0.35, 0.65)):
        self.name, self.k_varies, self.sigma = "membrane", False, sigma
        self.n, self.stride, self.d0, self.kappa = n, stride, d0, kappa          # kept for reporting (paper/make_numbers.py)
        mesh = rectangle_mesh(1.0, 1.0, n, n)
        fes = FESystem(mesh, Quad4Laplace(), sparse=True)
        fes.assemble_stiffness(1.0)
        fes.assemble_mass(1.0)
        x = mesh.nodes
        edge = np.flatnonzero((x[:, 0] < 1e-9) | (x[:, 0] > 1 - 1e-9) | (x[:, 1] < 1e-9) | (x[:, 1] > 1 - 1e-9))
        fes.fix_dofs(edge, ["u"])
        free = np.asarray(fes.free_dofs)
        self._K = sp.csr_matrix(fes.K)[free][:, free]
        self._M = sp.csr_matrix(fes.M)[free][:, free]
        self.xy = x[free]
        self.n_dof = len(free)
        grid = np.rint(self.xy * n).astype(int)
        chosen = np.flatnonzero((grid[:, 0] % stride == 0) & (grid[:, 1] % stride == 0))
        self.B = selection_matrix(chosen, self.n_dof)
        self.n_contact = len(chosen)
        r2 = np.sum((self.xy[chosen] - 0.5) ** 2, axis=1)
        self._offset = -(d0 + kappa * r2)
        self.lo = np.array([load_range[0], centre_range[0], centre_range[0]])
        self.hi = np.array([load_range[1], centre_range[1], centre_range[1]])
        self.param_names = ("peak load", "centre x", "centre y")

    def stiffness(self, mu):
        return self._K

    def load(self, mu):
        r2 = np.sum((self.xy - np.array([mu[1], mu[2]])) ** 2, axis=1)
        return -(self._M @ (mu[0] * np.exp(-r2 / (2 * self.sigma ** 2))))

    def offset(self, mu):
        return self._offset

    def sample_outside(self, rng, n):
        return sample_shell(rng, n, self.lo, self.hi, load_axis=0, load_factor=OUT_LOAD_FACTOR, centre_axes=(1, 2))


# --------------------------------------------------------------------------------------------- elastic block

class ElasticBlock(ContactProblem):
    """Elastic block ``[0,1]^2 x [0,depth]``: lateral faces clamped, Gaussian pressure ``P`` downward on the top face,
    bottom nodes may touch the foundation ``z >= -(d0 + kappa ((x - xc)^2 + (y - yc)^2))``.

    Parameters ``(P, xc, yc)``; if ``layered``, ``(rho, P, xc, yc)`` with ``rho = E_upper / E_lower``.
    """

    def __init__(self, n_xy: int = 16, nz: int = 4, depth: float = 0.5, nu: float = 0.3, layered: bool = False,
                 d0: float = 0.02, kappa: float = 1.0, sigma: float = 0.3, load_range=(0.10, 0.60),
                 centre_range=(0.35, 0.65), ratio_range=(0.5, 2.0)):
        self.name = "block_layered" if layered else "block"
        self.k_varies, self.layered, self.d0, self.kappa = layered, layered, d0, kappa
        self.n_xy, self.nz, self.depth, self.nu, self.sigma = n_xy, nz, depth, nu, sigma   # kept for reporting
        mesh = box_mesh(1.0, 1.0, depth, n_xy, n_xy, nz)
        D = D_solid3d(Material(E=1.0, nu=nu, rho=1.0))
        if layered:
            zc = mesh.nodes[mesh.elements, 2].mean(axis=1)
            upper = zc > depth / 2
            fes_mesh = MultiBlockMesh(nodes=mesh.nodes, blocks={"lower": mesh.elements[~upper], "upper": mesh.elements[upper]}, dim=3)
            fes = FESystem(fes_mesh, {"lower": Hex8Solid3D(), "upper": Hex8Solid3D()}, sparse=True)
        else:
            fes = FESystem(mesh, Hex8Solid3D(), sparse=True)
        x = mesh.nodes
        side = np.flatnonzero((x[:, 0] < 1e-9) | (x[:, 0] > 1 - 1e-9) | (x[:, 1] < 1e-9) | (x[:, 1] > 1 - 1e-9))
        fes.fix_dofs(side, [0, 1, 2])
        free = np.asarray(fes.free_dofs)
        if layered:
            with warnings.catch_warnings():                  # re-assembly replaces K on purpose (one layer at a time)
                warnings.simplefilter("ignore", UserWarning)
                fes.assemble_stiffness({"lower": D, "upper": 0.0 * D})
                self._K_lower = sp.csr_matrix(fes.K)[free][:, free]
                fes.assemble_stiffness({"lower": 0.0 * D, "upper": D})
                self._K_upper = sp.csr_matrix(fes.K)[free][:, free]
        else:
            fes.assemble_stiffness(D)
            self._K = sp.csr_matrix(fes.K)[free][:, free]
        self.n_dof = len(free)
        position = -np.ones(3 * len(x), dtype=int)
        position[free] = np.arange(len(free))
        interior = np.setdiff1d(np.arange(len(x)), side)
        bottom = interior[x[interior, 2] < 1e-9]
        top = interior[x[interior, 2] > depth - 1e-9]
        self.xy_contact = x[bottom, :2]
        self.n_contact = len(bottom)
        self.B = selection_matrix(position[3 * bottom + 2], self.n_dof)
        self._top_dof = position[3 * top + 2]
        hxy = 1.0 / n_xy
        self._top_weight = np.exp(-np.sum((x[top, :2] - 0.5) ** 2, axis=1) / (2 * sigma ** 2)) * hxy * hxy
        lo = [load_range[0], centre_range[0], centre_range[0]]
        hi = [load_range[1], centre_range[1], centre_range[1]]
        if layered:
            lo, hi = [ratio_range[0]] + lo, [ratio_range[1]] + hi
        self.lo, self.hi = np.array(lo), np.array(hi)
        self._p0 = 1 if layered else 0
        self.param_names = (("stiffness ratio",) if layered else ()) + ("pressure", "foundation x", "foundation y")

    def stiffness(self, mu):
        return self._K_lower + mu[0] * self._K_upper if self.layered else self._K

    def load(self, mu):
        f = np.zeros(self.n_dof)
        f[self._top_dof] = -mu[self._p0] * self._top_weight
        return f

    def offset(self, mu):
        r2 = np.sum((self.xy_contact - np.array([mu[self._p0 + 1], mu[self._p0 + 2]])) ** 2, axis=1)
        return -(self.d0 + self.kappa * r2)

    def sample_outside(self, rng, n):
        p0 = self._p0
        extra = (0, (self.hi[0] * OUT_RATIO_FACTOR[0], self.hi[0] * OUT_RATIO_FACTOR[1])) if self.layered else None
        return sample_shell(rng, n, self.lo, self.hi, load_axis=p0, load_factor=OUT_LOAD_FACTOR,
                            centre_axes=(p0 + 1, p0 + 2), extra=extra)
