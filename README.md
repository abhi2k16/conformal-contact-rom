# Conformal Screening of Silent Failures in Reduced Contact Models

[![GitHub](https://img.shields.io/badge/GitHub-abhi2k16%2Fconformal--contact--rom-blue?logo=github)](https://github.com/abhi2k16/conformal-contact-rom)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.2.6-013243?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.15.3-8CAAE6?logo=scipy)](https://scipy.org/)

> **Can a cheap admissibility indicator of a reduced-order contact model be turned into a screen with a finite-sample guarantee on the miss rate of silent failures — and what does it cost?**

Self-contained research project. Full-order FEA and the reduced contact model come from the suite packages `fea_engine` and `rom_engine` (`..\..\computation-suite`); everything else lives here.

---

## Project Structure

```
conformal-screening-contact-rom/
├── contactscreen/
│   ├── problems.py       # Benchmarks: Membrane (Quad4Laplace), ElasticBlock (Hex8Solid3D)
│   ├── fom.py            # Full-order solve (solve_contact_static_sparse)
│   ├── bases.py          # POD and dual bases
│   ├── rom.py            # ContactROM wrapper and admissibility indicator
│   ├── audit.py          # Per-query audits: truth, reduced answers, errors, indicators
│   ├── config.py         # Benchmark presets
│   ├── conformal.py      # Screening rules: class-conditional, weighted, risk control, heuristic
│   ├── focus.py          # Failure-focused calibration sampling
│   └── environment.py    # Run metadata written into every result JSON
├── scripts/
│   ├── generate_audits.py
│   ├── run_coverage.py        # Experiments A, B, G
│   ├── run_outside.py         # Experiments C, G
│   ├── run_fallback.py        # Experiments D, E
│   ├── run_enriched.py        # Experiment F
│   ├── run_condensed.py       # Experiment H
│   ├── generate_features.py   # Feature generation for learned rules
│   ├── run_learned.py         # Experiment I
│   ├── run_learned_speedup.py # Experiment J
│   └── run_crossfit.py        # Experiment K
├── tests/                # 33 tests: solver, ROM, and simulation checks of every rule
└── results/
    ├── audits/           # *.npz — one file per replicate
    └── *.json            # One result file per experiment
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e ..\..\computation-suite\fea_engine -e ..\..\computation-suite\rom_engine -e .
```

---

## Reproduction

```bash
# Run tests
.venv\Scripts\python -m pytest

# Generate audits
.venv\Scripts\python scripts\generate_audits.py --problem membrane --seeds 0-7   # block: 0-7, layered: 0-3

# Screening experiments
.venv\Scripts\python scripts\run_coverage.py  --problem membrane                 # A, B, G  (add --indicator eta2 for v2)
.venv\Scripts\python scripts\run_outside.py   --problem membrane                 # C, G
.venv\Scripts\python scripts\run_enriched.py  --problem membrane                 # F
.venv\Scripts\python scripts\run_fallback.py  --problem block                    # D, E  — run alone (timings)
.venv\Scripts\python scripts\run_condensed.py --problem block                    # H, constant K only — run alone

# Learned rules
.venv\Scripts\python scripts\generate_features.py --problem membrane --seeds 0-7
.venv\Scripts\python scripts\run_learned.py        --problem membrane            # I
```

---

## Versions

| Package | Version |
|---|---|
| Python | 3.10.18 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| `fea_engine` | 1.0.1 |
| `rom_engine` | 0.2.0 |
