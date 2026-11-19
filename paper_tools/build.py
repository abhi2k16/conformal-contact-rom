"""Build the manuscript and report the checks of manuscript_plan.md Section 7 (step 5).

Runs make_numbers.py, then pdflatex, bibtex, pdflatex twice in ``paper/build/``, and reports LaTeX errors, undefined
references and citations, overfull boxes above 10 pt, pending markers, the page count and typed numbers
(check_numbers.py). Exit code 1 if there is an error or an undefined reference or citation.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\build.py
Output: ``paper/build/main.pdf`` and the log.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent                # the generator scripts
PAPER = TOOLS.parent / "paper"                          # manuscript sources and outputs
BUILD = PAPER / "build"
MAX_OVERFULL_PT = 10.0

log = logging.getLogger("build")


def run(cmd: list, cwd: Path) -> subprocess.CompletedProcess:
    log.info("running %s", " ".join(cmd))
    env = dict(os.environ, BIBINPUTS=str(PAPER) + os.pathsep)          # bibtex runs in build/, refs.bib is in paper/
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def latex_chain() -> str:
    """pdflatex, bibtex, pdflatex x2; returns the final log text."""
    BUILD.mkdir(exist_ok=True)
    for tool in ("pdflatex", "bibtex"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"{tool} not found on PATH")
    tex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={BUILD}", "main.tex"]
    first = run(tex, PAPER)
    if first.returncode != 0:
        log.error("first pdflatex pass failed; see %s", BUILD / "main.log")
    env_bib = run(["bibtex", "main"], BUILD)
    if "error" in env_bib.stdout.lower():
        log.warning("bibtex: %s", env_bib.stdout.strip().splitlines()[-1])
    for _ in range(2):
        run(tex, PAPER)
    return (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")


def report(text: str) -> int:
    errors = re.findall(r"^! .*", text, flags=re.M)
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page", text)))
    undef_cite = sorted(set(re.findall(r"Citation `([^']+)' on page", text)))
    overfull = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", text)]
    pages = re.search(r"Output written on .*?\((\d+) pages?", text)
    pending = sum(p.read_text(encoding="utf-8").count("\\pending{") for p in (PAPER / "sections").glob("*.tex"))
    pending += (PAPER / "main.tex").read_text(encoding="utf-8").count("\\pending{")
    log.info("LaTeX errors: %d %s", len(errors), errors[:3])
    log.info("undefined references: %d %s", len(undef_ref), undef_ref)
    log.info("undefined citations: %d %s", len(undef_cite), undef_cite)
    log.info("overfull boxes > %.0f pt: %d", MAX_OVERFULL_PT, sum(x > MAX_OVERFULL_PT for x in overfull))
    log.info("pending markers: %d", pending)
    log.info("pages: %s", pages.group(1) if pages else "none (no PDF)")
    return 1 if errors or undef_ref or undef_cite or not pages else 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    python = sys.executable
    for script in ("make_numbers.py", "make_tables.py", "make_figures.py"):
        if run([python, str(TOOLS / script)], PAPER.parent).returncode != 0:
            log.error("%s failed", script)
            sys.exit(1)
    status = report(latex_chain())
    checked = run([python, str(TOOLS / "check_numbers.py")], PAPER.parent)
    print(checked.stdout.strip())
    sys.exit(status or checked.returncode)


if __name__ == "__main__":
    main()
