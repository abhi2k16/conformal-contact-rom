"""List digits typed in the manuscript text instead of generated (AUDIT_GUIDELINE.md §8; manuscript_plan.md Section 6).

Scans ``paper/sections/*.tex`` after removing comments, ``\\R{...}``, labels, references, citations, environments
and commands whose arguments are not prose, and inline or display math. Whatever digits remain are reported with file
and line, so a typed result number is caught. Some hits are legitimate (a fixed confidence level such as "95",
"one thread"); they are listed for the author to judge, and the exit code is 0 unless ``--strict`` is given.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\check_numbers.py [--strict]
"""

import argparse
import re
import sys
from pathlib import Path

SECTIONS = Path(__file__).resolve().parents[1] / "paper" / "sections"
STRIP = [
    r"(?<!\\)%.*$",                                   # comments
    r"\\(R|label|ref|cref|Cref|eqref|[Cc]ite[a-z]*|pending|begin|end|includegraphics)\*?(\[[^\]]*\])*\{[^{}]*\}",
    r"\$\$.*?\$\$", r"\\\[.*?\\\]", r"\$[^$]*\$",     # math
    r"\\(textbf|emph|texttt)\b",                      # keep argument text, drop the command name
    r"\b(NMF|SVD)(~|\s)?\d+",                          # model names (dual-basis size), not results
    r"\\(setlength|input)\{[^{}]*\}\{?[^{}]*\}?",      # layout commands and file names
]
ALLOWED = {"95", "90"}                                   # fixed choices of the protocol, not results


def typed_numbers(text: str) -> list:
    hits = []
    in_math_env = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if re.search(r"\\begin\{(equation|align|tabular)", line):
            in_math_env = True
        if in_math_env:
            if re.search(r"\\end\{(equation|align|tabular)", line):
                in_math_env = False
            continue
        clean = line
        for pattern in STRIP:
            clean = re.sub(pattern, " ", clean)
        for match in re.finditer(r"(?<![A-Za-z\\])\d+(?:[.,]\d+)?", clean):
            if match.group() not in ALLOWED:
                hits.append((lineno, match.group(), line.strip()[:100]))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="exit with 1 if any typed number is found")
    args = parser.parse_args()
    total = 0
    for path in sorted(SECTIONS.glob("*.tex")):
        for lineno, num, context in typed_numbers(path.read_text(encoding="utf-8")):
            print(f"typed number {num!r} in {path.name}:{lineno}: {context}")
            total += 1
    print(f"typed numbers found: {total}")
    sys.exit(1 if args.strict and total else 0)


if __name__ == "__main__":
    main()
