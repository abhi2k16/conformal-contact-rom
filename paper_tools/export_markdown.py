"""Export the abstract and Sections 1 to 5 of the LaTeX manuscript to ``introduction_and_methodology.md``.

The Markdown file is a generated copy of the LaTeX text, not a separate draft. The export
  * replaces every ``\\R{key}`` by its value from ``paper/numbers.tex``;
  * replaces ``\\cref``/``\\Cref``/``\\eqref`` by the numbers of the built PDF (``paper/build/main.aux``) and adds
    ``\\tag{n}`` to numbered equations;
  * replaces ``\\citep``/``\\citet`` by the IEEE numbers of the built PDF (``paper/build/main.bbl``), with the author
    names for ``\\citet``, and appends the cited references in that numbering;
  * writes Algorithm 1 as an indented list (pandoc drops ``algorithmic``) and links the figure files;
  * converts the rest with pandoc (LaTeX to Markdown with TeX math), expanding the macros of ``macros.tex``.
Run ``build.py`` first so that numbers, labels and the bibliography are current.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\export_markdown.py
Output: ``introduction_and_methodology.md`` (overwritten).
"""

import datetime
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "introduction_and_methodology.md"
SECTIONS = ("01_introduction", "02_model", "03_indicators", "04_thresholds", "05_benchmarks")
LABEL_TYPES = {"sec": ("Section", "Sections"), "app": ("Appendix", "Appendices"), "tab": ("Table", "Tables"),
               "fig": ("Figure", "Figures"), "alg": ("Algorithm", "Algorithms")}

log = logging.getLogger("export_markdown")


def read_numbers() -> dict:
    text = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\csname res@(.*?)\\endcsname\{(.*)\}", text))


def read_labels() -> dict:
    aux = (PAPER / "build" / "main.aux").read_text(encoding="utf-8", errors="replace")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}", aux)}


def read_bibliography() -> tuple:
    """({key: number}, {key: short author text}) in the order of the built bibliography."""
    bbl = (PAPER / "build" / "main.bbl").read_text(encoding="utf-8", errors="replace")
    numbers, authors = {}, {}
    for i, m in enumerate(re.finditer(r"\\bibitem\[(.*?)\]\{([^}]+)\}", bbl, flags=re.S), start=1):
        short, key = m.group(1), m.group(2)
        numbers[key] = i
        authors[key] = latex_text(short.split("(")[0].strip())
    return numbers, authors


def latex_text(s: str) -> str:
    """Plain text of a short BibTeX string (accents, ties)."""
    accents = {r"{\`e}": "è", r"{\'e}": "é", r'{\"u}': "ü", r"{\'u}": "ú", r"{\`a}": "à", "~": " ", "{": "", "}": ""}
    for a, b in accents.items():
        s = s.replace(a, b)
    return " ".join(s.split())


def ref_text(labels: list, table: dict, capital: bool) -> str:
    kinds = {lab.split(":")[0] for lab in labels}
    nums = [table[lab] for lab in labels]
    if kinds == {"eq"}:
        return ", ".join(f"({n})" for n in nums) if len(nums) > 1 else f"({nums[0]})"
    kind = kinds.pop() if len(kinds) == 1 else "sec"
    single, plural = LABEL_TYPES.get(kind, ("", ""))
    word = single if len(nums) == 1 else plural
    joined = nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]
    return f"{word} {joined}"


def substitute(tex: str, numbers: dict, labels: dict, bib: tuple) -> tuple:
    cited = set()
    bib_numbers, bib_authors = bib

    def r_value(m):
        if m.group(1) not in numbers:
            raise KeyError(f"unknown number key {m.group(1)}")
        return numbers[m.group(1)]

    tex = re.sub(r"\\R\{([^}]+)\}", r_value, tex)

    def cref(m):
        labs = [s.strip() for s in m.group(2).split(",")]
        missing = [lab for lab in labs if lab not in labels]
        if missing:
            raise KeyError(f"unknown labels {missing}")
        return ref_text(labs, labels, m.group(1) == "C")

    tex = re.sub(r"\\([cC])ref\{([^}]+)\}", cref, tex)
    tex = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({labels[m.group(1)]})", tex)

    def cite(m):
        keys = [k.strip() for k in m.group(3).split(",")]
        cited.update(keys)
        nums = ", ".join(str(bib_numbers[k]) for k in keys)
        if m.group(1).lower() == "citet":
            return f"{bib_authors[keys[0]]} [{nums}]"
        return f"[{nums}]"

    tex = re.sub(r"\\([Cc]ite[pt]|cite)(\[[^\]]*\])?\{([^}]+)\}", cite, tex)

    def equation(m):                                    # numbered equations keep their number as a tag
        body = m.group(1)
        lab = re.search(r"\\label\{([^}]+)\}", body)
        if lab:
            body = body.replace(lab.group(0), f"\\tag{{{labels[lab.group(1)]}}}")
        return f"\\begin{{equation*}}{body}\\end{{equation*}}"

    tex = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", equation, tex, flags=re.S)
    return tex, cited


ALG_TOKEN = "ALGORITHMPLACEHOLDER"


def braced(s: str, start: int) -> tuple:
    """Content of the brace group opening at ``s[start] == '{'`` and the index after it."""
    if s[start] != "{":
        raise ValueError("expected '{'")
    depth = 0
    for i in range(start, len(s)):
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        if depth == 0:
            return s[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def replace_command(s: str, name: str, render) -> str:
    """Replace every ``\\name{arg}`` by ``render(arg)`` (balanced braces)."""
    out, i = "", 0
    while True:
        j = s.find("\\" + name + "{", i)
        if j < 0:
            return out + s[i:]
        arg, end = braced(s, j + len(name) + 1)
        out, i = out + s[i:j] + render(arg), end


def simple_macros() -> dict:
    """Argument-free macros of macros.tex (\\newcommand{\\x}{...}), for math that pandoc does not see."""
    out = {}
    for line in (PAPER / "macros.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
        if m and not line.startswith(("\\newcommand{\\R}", "\\newcommand{\\pending}")):
            out[m.group(1)] = m.group(2)
    return out


def expand(s: str, macros: dict) -> str:
    for _ in range(3):                                   # macros may use other macros
        s = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", lambda m: macros.get(m.group(1), m.group(0)), s)
    return s


def algorithm_markdown(tex: str, labels: dict) -> tuple:
    """Replace the algorithm float by a placeholder; return (tex, Markdown list) (pandoc drops algorithmic)."""
    m = re.search(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", tex, flags=re.S)
    if m is None:
        return tex, ""
    block, macros = m.group(0), simple_macros()
    cap_start = block.index("\\caption{") + len("\\caption")
    caption, _ = braced(block, cap_start)
    label = re.search(r"\\label\{(alg:[^}]+)\}", block).group(1)
    out, depth = [f"**Algorithm {labels[label]}.** {expand(caption, macros)}", ""], 0
    body = block[block.index("\\begin{algorithmic}"):block.index("\\end{algorithmic}")]
    for line in body.split("\n")[1:]:
        s = line.strip().replace("\\label{alg:fallback}", "")
        if not s:
            continue
        if s.startswith("\\EndIf"):
            depth -= 1
            continue
        opens = s.startswith("\\If{") and "\\EndIf" not in s
        s = replace_command(s, "Comment", lambda a: f" — *{a}*")
        s = replace_command(s, "If", lambda a: f"**if** {a} **then**")
        s = replace_command(s, "textbf", lambda a: a)
        s = re.sub(r"\\Require\s*", "**Require:** ", s)
        s = s.replace("\\Return", "**return**").replace("\\State", "").replace("\\EndIf", "")
        s = re.sub(r"line~?\\ref\{alg:fallback\}", "the warm-started fallback (last two steps)", s)
        out.append("    " * depth + "- " + " ".join(expand(s, macros).split()))
        depth += int(opens)
    return tex[:m.start()] + "\n\n" + ALG_TOKEN + "\n\n" + tex[m.end():], "\n".join(out)


def macros_for_pandoc() -> str:
    """The notation macros of macros.tex, without the \\R and \\pending helpers."""
    lines = (PAPER / "macros.tex").read_text(encoding="utf-8").splitlines()
    keep = [l for l in lines if l.startswith("\\newcommand") and not l.startswith(("\\newcommand{\\R}", "\\newcommand{\\pending}"))]
    return "\n".join(keep)


def inline_tables(tex: str) -> str:
    return re.sub(r"\\input\{(tables/[^}]+)\}", lambda m: (PAPER / (m.group(1) + ".tex")).read_text(encoding="utf-8"), tex)


def pandoc(tex: str) -> str:
    if shutil.which("pandoc") is None:
        raise FileNotFoundError("pandoc not found on PATH")
    run = subprocess.run(["pandoc", "-f", "latex", "-t", "markdown-raw_tex+tex_math_dollars", "--wrap=none"],
                         input=tex, capture_output=True, text=True, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(f"pandoc failed: {run.stderr[:500]}")
    return run.stdout


def references(cited: set, bib_numbers: dict) -> str:
    """Cited references in IEEE numbering, from the built bibliography (plain text)."""
    bbl = (PAPER / "build" / "main.bbl").read_text(encoding="utf-8", errors="replace")
    items = re.split(r"\\bibitem\[.*?\]\{([^}]+)\}", bbl, flags=re.S)
    entries = {items[i]: items[i + 1] for i in range(1, len(items) - 1, 2)}
    out = ["## References", ""]
    for key in sorted(cited, key=lambda k: bib_numbers[k]):
        body = entries[key].split("\\end{thebibliography}")[0]
        body = re.sub(r"\\BIBentry\w*\s*", "", body)
        body = re.sub(r"\\(emph|textit)\{([^}]*)\}", r"*\2*", body)
        body = re.sub(r"\\url\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\\[A-Za-z]+\s*", " ", body)
        out.append(f"[{bib_numbers[key]}] {latex_text(body).replace(' ,', ',').replace(' .', '.')}")
        out.append("")
    return "\n".join(out)


def tidy(md: str) -> str:
    """Typographic clean-up of pandoc and bibliography output (headings, dashes, quotes, citation brackets)."""
    md = re.sub(r" \{#[^}]*\}", "", md)                                  # pandoc heading attributes
    md = re.sub(r"(?<=[\w)])--(?=[\w(])", "\u2013", md)                 # en dashes in words and page ranges
    md = md.replace("``", "\u201c").replace("''", "\u201d")
    md = re.sub(r"1em plus 0\.5em minus 0\.4em\s*", "", md)             # \newblock spacing of IEEEtran
    md = re.sub(r"\\\[([0-9][0-9, \u2013]*)\\\]", r"[\1]", md)        # \[3, 4\] -> [3, 4]
    return md


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for needed in (PAPER / "numbers.tex", PAPER / "build" / "main.aux", PAPER / "build" / "main.bbl"):
        if not needed.exists():
            raise FileNotFoundError(f"{needed} missing; run paper_tools/build.py first")
    numbers, labels, bib = read_numbers(), read_labels(), read_bibliography()
    title = re.search(r"\\title\{(.*?)\}", (PAPER / "main.tex").read_text(encoding="utf-8"), flags=re.S).group(1)
    parts = ["\\section*{Abstract}\n" + (PAPER / "sections" / "00_abstract.tex").read_text(encoding="utf-8")]
    parts += [(PAPER / "sections" / f"{name}.tex").read_text(encoding="utf-8") for name in SECTIONS]
    tex, cited = substitute(inline_tables("\n\n".join(parts)), numbers, labels, bib)
    tex, alg_md = algorithm_markdown(tex, labels)
    tex = re.sub(r"\\includegraphics\{figures/([^}]+)\.pdf\}", r"\\includegraphics{paper/figures/\1.png}", tex)
    tex = re.sub(r"\\section\{([^}]*)\}\\label\{([^}]+)\}", lambda m: f"\\section*{{{labels[m.group(2)]}. {m.group(1)}}}", tex)
    tex = re.sub(r"\\subsection\{([^}]*)\}\\label\{([^}]+)\}", lambda m: f"\\subsection*{{{labels[m.group(2)]} {m.group(1)}}}", tex)
    body = pandoc(macros_for_pandoc() + "\n\n" + tex).replace(ALG_TOKEN, alg_md)
    for kind, word in (("fig", "Figure"), ("tab", "Table")):
        for lab, num in labels.items():
            if lab.startswith(kind + ":"):
                body = body.replace(f'{{#{lab}}}', "")
    header = (f"# {title}\n\n*Generated on {datetime.date.today().isoformat()} from the LaTeX manuscript (`paper/`, abstract and "
              "Sections 1 to 5) by `paper_tools/export_markdown.py`. Edit the LaTeX sources, not this file; numbers, "
              "references and citations follow the built PDF.*\n\n")
    OUT.write_text(header + tidy(body.strip()) + "\n\n" + tidy(references(cited, bib[0])) + "\n", encoding="utf-8")
    log.info("wrote %s (%d characters, %d references)", OUT, OUT.stat().st_size, len(cited))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
