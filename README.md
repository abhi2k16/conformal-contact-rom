# Conformal screening of silent failures in reduced contact models

[![GitHub](https://img.shields.io/badge/GitHub-abhi2k16%2Fconformal--contact--rom-blue?logo=github)](https://github.com/abhi2k16/conformal-contact-rom)

Self-contained research project. Question: can a cheap admissibility indicator of a reduced-order contact model be turned into a screen with a finite-sample guarantee on the miss rate of silent failures, and what does it cost? Plan, hypotheses, results and status: [research_plan.md](research_plan.md). Coding rules: [SCRIPT_GUIDELINES.md](SCRIPT_GUIDELINES.md).

No code or data from another project is used. Full-order FEA and the reduced contact model come from the suite packages `fea_engine` and `rom_engine` (`..\..\computation-suite`); everything else is in this folder.

## Layout

| Path | Content |
|---|---|
| `contactscreen/problems.py` | benchmarks: `Membrane` (`Quad4Laplace`), `ElasticBlock` (`Hex8Solid3D`, optional two layers so that `K` varies) |
| `contactscreen/fom.py`, `bases.py`, `rom.py` | full-order solve (`solve_contact_static_sparse`), POD and dual bases, `ContactROM` wrapper and the admissibility indicator |
| `contactscreen/audit.py`, `config.py` | per-query audits (truth, reduced answers, errors, indicators) and the benchmark presets |
| `contactscreen/conformal.py`, `focus.py` | screening rules (class-conditional, weighted, risk control, heuristic) and failure-focused calibration sampling |
| `scripts/` | `generate_audits.py`, `run_coverage.py` (A, B, G), `run_outside.py` (C, G), `run_fallback.py` (D, E), `run_enriched.py` (F), `run_condensed.py` (H), `generate_features.py` + `run_learned.py` (I), `run_learned_speedup.py` (J), `run_crossfit.py` (K) |
| `contactscreen/environment.py` | run metadata (CPU, versions, BLAS, threads, direct solver) written into every result JSON |
| `tests/` | 33 tests: solver and ROM checks against independent methods, simulation checks of every rule |
| `results/` | `audits/*.npz` (one per replicate) and one JSON per experiment |

## Setup and reproduction

    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
    .venv\Scripts\pip install -e ..\..\computation-suite\fea_engine -e ..\..\computation-suite\rom_engine -e .
    .venv\Scripts\python -m pytest
    .venv\Scripts\python scripts\generate_audits.py --problem membrane --seeds 0-7    # block 0-7, layered 0-3
    .venv\Scripts\python scripts\run_coverage.py --problem membrane                  # add --indicator eta2 for indicator v2
    .venv\Scripts\python scripts\run_outside.py  --problem membrane
    .venv\Scripts\python scripts\run_enriched.py --problem membrane
    .venv\Scripts\python scripts\run_fallback.py --problem block                       # timings: run alone
    .venv\Scripts\python scriptsun_condensed.py --problem block                      # H, constant K only; run alone
    .venv\Scripts\python scripts\generate_features.py --problem membrane --seeds 0-7    # then run_learned.py (I)
    .venv\Scripts\python scriptsun_learned.py --problem membrane

Versions used: Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, `fea_engine` 1.0.1, `rom_engine` 0.2.0.
