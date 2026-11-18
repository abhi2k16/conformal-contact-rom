"""Benchmark presets: problem constructors and audit settings, in one place.

Settings were chosen so that the reduced models span a range of silent-failure rates (rare to frequent); see
``research_plan_conformal_screening.md``, Section 5.
"""

from pathlib import Path

from .audit import AuditSettings
from .problems import ContactProblem, ElasticBlock, Membrane

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT_DIR = RESULTS / "audits"

# Seeds of every random choice outside the audit replicates (replicate seeds are the audit file seeds).
SEEDS = {
    "split": 0,             # random calibration/test splits (run_coverage, run_outside, calibration subsets of run_fallback)
    "bootstrap": 1,         # bootstrap over replicates
    "enriched": 0,          # calibration draws of run_enriched
    "timing_queries": 10_000,   # run_fallback draws its fresh queries with seed timing_queries + replicate seed
}

PROBLEMS = {
    "membrane": lambda: Membrane(),
    "block": lambda: ElasticBlock(n_xy=12, nz=3),
    "layered": lambda: ElasticBlock(n_xy=16, nz=4, layered=True),
}

SETTINGS = {
    "membrane": dict(n_train=200, r_primal=60, n_pool=800, n_outside=300,
                     models=(("svd", 24), ("nmf", 12), ("nmf", 24), ("nmf", 36), ("nmf", 48))),
    "block": dict(n_train=100, r_primal=30, n_pool=400, n_outside=200,
                  models=(("svd", 12), ("nmf", 8), ("nmf", 12), ("nmf", 16), ("nmf", 24))),
    "layered": dict(n_train=100, r_primal=30, n_pool=300, n_outside=150,
                    models=(("svd", 12), ("nmf", 8), ("nmf", 12), ("nmf", 16), ("nmf", 24))),
}


def make_problem(name: str) -> ContactProblem:
    if name not in PROBLEMS:
        raise ValueError(f"unknown problem {name!r}; choose from {sorted(PROBLEMS)}")
    return PROBLEMS[name]()


def make_settings(name: str, seed: int) -> AuditSettings:
    if name not in SETTINGS:
        raise ValueError(f"unknown problem {name!r}; choose from {sorted(SETTINGS)}")
    return AuditSettings(seed=seed, **SETTINGS[name])


def audit_path(name: str, seed: int) -> Path:
    return AUDIT_DIR / f"{name}_seed{seed}.npz"
