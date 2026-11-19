"""Look up generated numbers while drafting: print the value of each key, or of every key matching a pattern.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\lookup_numbers.py KEY [KEY ...]
    .venv\\Scripts\\python paper_tools\\lookup_numbers.py --grep "crossfit.block.nmf.8.240.*cc0.1"
Reads ``paper/numbers.tex`` (run ``make_numbers.py`` first). Exit code 1 if a named key is missing.
"""

import argparse
import re
import sys
from pathlib import Path

NUMBERS = Path(__file__).resolve().parents[1] / "paper" / "numbers.tex"
ENTRY = re.compile(r"\\csname res@(.*?)\\endcsname\{(.*)\}")


def read_numbers() -> dict:
    if not NUMBERS.exists():
        raise FileNotFoundError(f"{NUMBERS} missing; run make_numbers.py")
    return dict(ENTRY.findall(NUMBERS.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--grep", help="regular expression matched against the keys")
    args = parser.parse_args()
    numbers = read_numbers()
    missing = 0
    for key in args.keys:
        value = numbers.get(key)
        missing += value is None
        print(f"{key} = {value if value is not None else 'MISSING'}")
    if args.grep:
        pattern = re.compile(args.grep)
        for key in sorted(k for k in numbers if pattern.search(k)):
            print(f"{key} = {numbers[key]}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
