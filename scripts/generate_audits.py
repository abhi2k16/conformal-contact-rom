"""Generate audit replicates: train reduced models, evaluate in-range and out-of-range query pools, save ``.npz``.

Each replicate (seed) is an independent run of the whole pipeline (training parameters, bases, query pools).
    python scripts/generate_audits.py --problem membrane --seeds 0-7
Output: ``results/audits/<problem>_seed<k>.npz``. Existing files are kept unless ``--overwrite`` is given.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contactscreen.audit import run_audit, save_audit                     # noqa: E402
from contactscreen.config import PROBLEMS, audit_path, make_problem, make_settings   # noqa: E402

log = logging.getLogger("generate_audits")


def parse_seeds(text: str) -> list:
    """``"0-7"`` -> [0..7]; ``"1,3,5"`` -> [1, 3, 5]."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    problem = make_problem(args.problem)
    for seed in parse_seeds(args.seeds):
        path = audit_path(args.problem, seed)
        if path.exists() and not args.overwrite:
            log.info("%s exists, skipping", path.name)
            continue
        start = time.perf_counter()
        audit = run_audit(problem, make_settings(args.problem, seed))
        save_audit(audit, path)
        log.info("%s done in %.0f s", path.name, time.perf_counter() - start)


if __name__ == "__main__":
    main()
