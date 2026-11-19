"""Write the manuscript tables to ``paper/tables/*.tex`` from generated keys (manuscript_plan.md Sections 4 and 6).

Every cell is a ``\\R{key}`` reference, so the tables use the same rounding as the text. Each key is checked against
``paper/numbers.tex`` (run ``make_numbers.py`` first); a missing key stops the script.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\make_tables.py
Outputs: ``tables/rates.tex`` (Section 7), ``tables/equal_budget.tex`` (Section 6), ``tables/layered_speedup.tex`` (Section 9).
"""

import logging
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
TABLES = PAPER / "tables"
PROBLEMS = (("membrane", "Membrane"), ("block", "Block"), ("layered", "Layered block"))
CROSSFIT_T = {"membrane": 400, "block": 240, "layered": 160}          # the larger budget of run_crossfit.py
CROSSFIT_MODELS = {"membrane": (24, 36, 48), "block": (8, 12, 16, 24), "layered": (8, 12, 16, 24)}

log = logging.getLogger("make_tables")


def known_keys() -> set:
    text = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    return set(re.findall(r"\\csname res@(.*?)\\endcsname", text))


class Cells:
    """``R(key)`` returns ``\\R{key}`` after checking the key exists."""

    def __init__(self):
        self.keys = known_keys()

    def __call__(self, key: str) -> str:
        if key not in self.keys:
            raise KeyError(f"no generated number {key!r}; run make_numbers.py or fix the key")
        return f"\\R{{{key}}}"


def rates_table(R: Cells) -> str:
    rows = []
    for name, label in PROBLEMS:
        keys = {k.split(".")[3] + "|" + k.split(".")[4] for k in R.keys
                if k.startswith(f"coverage.{name}.rates.") and k.endswith(".p")}
        order = sorted(keys, key=lambda s: (s.split("|")[0] != "svd", int(s.split("|")[1])))   # SVD control first
        for i, mk in enumerate(order):
            kind, r = mk.split("|")
            pre = f"coverage.{name}.rates.{kind}.{r}"
            auc = R(f"{pre}.auc") if f"{pre}.auc" in R.keys else "--"
            need = lambda a: R(f"{pre}.queries_needed@{a}") if f"{pre}.queries_needed@{a}" in R.keys else "--"
            first = label if i == 0 else ""
            rows.append(f"{first} & {kind.upper()} {r} & {R(pre + '.p')} & [{R(pre + '.p_ci.0')}, {R(pre + '.p_ci.1')}] & "
                        f"{auc} & {need('0.05')} & {need('0.1')} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    return ("\\begin{tabular}{llrcrrr}\n\\toprule\n"
            "Benchmark & Dual basis & $p$ & 95\\,\\% interval & AUC of $\\eta$ & queries, $\\alpha = 0.05$ & queries, $\\alpha = 0.1$ \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\end{tabular}\n")


def equal_budget_table(R: Cells, alpha: str) -> str:
    rows = []
    for name, label in PROBLEMS:
        T = CROSSFIT_T[name]
        for i, r in enumerate(CROSSFIT_MODELS[name]):
            cells = []
            for variant in ("eta", "split", "xconf"):
                pre = f"crossfit.{name}.nmf.{r}.{T}.{variant}.cc{alpha}"
                cells.append(f"{R(pre + '.miss')} & {R(pre + '.fpr')} & {R(pre + '.defined')}")
            first = f"{label}, $T = {T}$" if i == 0 else ""
            rows.append(f"{first} & NMF {r} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    head = ("\\begin{tabular}{ll" + "rrr" * 3 + "}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{$\\eta$} & \\multicolumn{3}{c}{learned, split} & \\multicolumn{3}{c}{learned, cross-fitted} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
            "Benchmark & Model & miss & false al. & def. & miss & false al. & def. & miss & false al. & def. \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\end{tabular}\n"


def layered_speedup_table(R: Cells) -> str:
    rows = []
    for r in (8, 12, 16, 24):
        pre = f"learned.speedup.layered.models.nmf.{r}"
        cells = [R(f"{pre}.silent_rate")]
        for variant in ("eta@160", "xconf@160"):
            c = f"{pre}.{variant}.cc0.1"
            cells += [f"{R(c + '.speedup_cold')} / {R(c + '.speedup_warm')}", R(c + ".miss_rate"), R(c + ".fallback")]
        cells.append(f"{R(pre + '.oracle.speedup_cold')} / {R(pre + '.oracle.speedup_warm')}")
        rows.append(f"NMF {r} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{$\\eta$, $\\cc(0.1)$} & \\multicolumn{3}{c}{cross-fitted, $\\xconf(0.1)$} & oracle \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Model & $p$ & speedup & miss & fallback & speedup & miss & fallback & speedup \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def eps_table(R: Cells) -> str:
    """Failure rate, AUC of eta and calibration queries at alpha 0.05 for three error tolerances, every NMF model."""
    tolerances = ("0.02", "0.05", "0.1")
    rows = []
    for name, label in PROBLEMS:
        sizes = sorted({int(m.group(1)) for k in R.keys if (m := re.match(rf"eps\.{name}\.nmf\.(\d+)\.", k))})
        for i, r in enumerate(sizes):
            kind = "nmf"
            cells = []
            for tol in tolerances:
                pre = f"eps.{name}.{kind}.{r}.{tol}"
                auc = R(f"{pre}.auc") if f"{pre}.auc" in R.keys else "--"
                need = R(f"{pre}.need@0.05") if f"{pre}.need@0.05" in R.keys else "--"
                cells += [R(f"{pre}.p"), auc, need]
            rows.append((label if i == 0 else "") + f" & NMF {r} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    head = ("\\begin{tabular}{ll" + "rrr" * 3 + "}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{$\\varepsilon = \\R{sum.E.eps.lo}$} & \\multicolumn{3}{c}{$\\varepsilon = \\R{sum.E.eps.mid}$} & "
            "\\multicolumn{3}{c}{$\\varepsilon = \\R{sum.E.eps.hi}$} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
            "Benchmark & Model & $p$ & AUC & queries & $p$ & AUC & queries & $p$ & AUC & queries \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\end{tabular}\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (PAPER / "numbers.tex").exists():
        raise FileNotFoundError("paper/numbers.tex missing; run make_numbers.py first")
    TABLES.mkdir(exist_ok=True)
    R = Cells()
    outputs = {"rates.tex": rates_table(R), "equal_budget.tex": equal_budget_table(R, "0.1"),
               "equal_budget_005.tex": equal_budget_table(R, "0.05"), "layered_speedup.tex": layered_speedup_table(R),
               "eps.tex": eps_table(R)}
    for name, text in outputs.items():
        (TABLES / name).write_text("% Generated by paper_tools/make_tables.py. Do not edit.\n" + text, encoding="utf-8")
        log.info("wrote %s", TABLES / name)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, FileNotFoundError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
