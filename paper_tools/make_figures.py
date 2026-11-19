"""Draw the manuscript figures from ``results/`` only (no solves), sized for the page (REPORT_PUBLISHING_GUIDELINE.md §3).

Figures are drawn at the text width ``W_IN`` with 10 pt labels, 9 pt ticks and 8.5 pt legends and saved as vector PDF
in ``paper/figures/``. Hidden series are avoided with open markers and distinct line styles.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\make_figures.py
Outputs: ``figures/silent_failure.pdf`` (Section 2), ``figures/roc_curves.pdf`` (Section 6), from the data written by
``make_figure_data.py``; ``figures/calibration_cost.pdf`` (Section 7), ``figures/outside_tradeoff.pdf`` (Section 8),
``figures/query_cost.pdf`` and ``figures/layered_speedup.pdf`` (Section 9).
"""

import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
W_IN = 6.3                                     # text width of main.tex (A4, 25 mm margins = 160 mm); guideline value 6.9 in is for its own layout
SINGLE = 0.78 * W_IN                           # single-panel figures
STYLE = {"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
         "legend.fontsize": 8.5, "savefig.bbox": "tight", "pdf.fonttype": 42}
PROBLEMS = (("membrane", "Membrane", "o"), ("block", "Block", "s"), ("layered", "Layered block", "^"))
COLORS = {"membrane": "#1f77b4", "block": "#d62728", "layered": "#2ca02c"}

log = logging.getLogger("make_figures")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    log.info("wrote %s", FIGURES / name)


def calibration_cost() -> None:
    """Calibration queries needed by cc(0.05) against the silent-failure rate p, every model, with n_min / p."""
    from contactscreen.conformal import min_failures

    n_min = min_failures(0.05)
    fig, ax = plt.subplots(figsize=(SINGLE, 0.62 * SINGLE))
    ps = [10 ** (x / 50) for x in range(-100, 1)]
    ax.plot(ps, [n_min / p for p in ps], color="0.5", lw=1.2, ls="--", label=f"$n_{{\\min}}/p$, $n_{{\\min}} = {n_min}$")
    for name, label, marker in PROBLEMS:
        rates = load(f"coverage_{name}.json")["rates"]
        x = [r["p"] for r in rates.values()]
        y = [r["queries_needed@0.05"] for r in rates.values()]
        xerr = [[r["p"] - r["p_ci"][0] for r in rates.values()], [r["p_ci"][1] - r["p"] for r in rates.values()]]
        ax.errorbar(x, y, xerr=xerr, fmt=marker, mfc="none", color=COLORS[name], ms=6, capsize=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("silent-failure rate $p$")
    ax.set_ylabel("calibration queries for $\\mathrm{cc}(0.05)$")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, which="major", lw=0.4, alpha=0.5)
    save(fig, "calibration_cost.pdf")


def outside_tradeoff() -> None:
    """Out of range: miss rate against fallback share as the parameter margin shrinks (most accurate model each)."""
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 0.42 * W_IN), sharey=True)
    for ax, rule, title in ((axes[0], "cc0.1", "$\\mathrm{cc}(0.1)$"), (axes[1], "crc0.01", "$\\mathrm{crc}(0.01)$")):
        for name, label, marker in PROBLEMS:
            out = load(f"outside_{name}.json")
            model = "nmf|48" if name == "membrane" else "nmf|24"
            margins = ["none"] + [str(m) for m in out["margins"]]
            pts = [out["cells"][f"{model}|{rule}|{m}"] for m in margins]
            ax.plot([c["flagged"] for c in pts], [c["miss"] for c in pts], marker=marker, mfc="none", color=COLORS[name],
                    label=f"{label}, NMF {model.split('|')[1]}")
            ax.plot(pts[0]["flagged"], pts[0]["miss"], marker=marker, color=COLORS[name], ms=8, ls="none")   # no guard: filled
        ax.set_title(title)
        ax.set_xlabel("fallback share")
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[0].set_ylabel("miss rate (out of range)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save(fig, "outside_tradeoff.pdf")


def query_cost() -> None:
    """Median time per query: reduced solve with eta, cold and warm suite solve, condensed solve (constant K)."""
    labels, groups = [], []
    for name, label, _ in PROBLEMS:
        fb = load(f"fallback_{name}.json")["D"]
        model = "nmf|48" if name == "membrane" else "nmf|24"
        m = fb["models"][model]
        warm = next(r for r in m["by_rel"] if r["rel"] == m["chosen_rel"])["t_warm_ms_median"]
        row = {"reduced solve + $\\eta$": m["t_rom_ms_median"], "full order, warm": warm,
               "full order, cold": fb["fom"]["t_cold_ms_median"]}
        if name != "layered":
            row["condensed full order"] = load(f"condensed_{name}.json")["H"]["condensed_ms_median"]
        labels.append(f"{label}\n(NMF {model.split('|')[1]})")
        groups.append(row)
    kinds = ["reduced solve + $\\eta$", "condensed full order", "full order, warm", "full order, cold"]
    colors = ["#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]
    fig, ax = plt.subplots(figsize=(W_IN, 0.42 * W_IN))
    width = 0.2
    for j, (kind, color) in enumerate(zip(kinds, colors)):
        xs = [i + (j - 1.5) * width for i, g in enumerate(groups) if kind in g]
        ys = [g[kind] for g in groups if kind in g]
        ax.bar(xs, ys, width, color=color, label=kind)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("median time per query (ms)")
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    save(fig, "query_cost.pdf")


def layered_speedup() -> None:
    """Net speedup on the layered block, cc(0.1): eta against the cross-fitted indicator, warm fallback, with the oracle."""
    models = load("learned_speedup_layered.json")["models"]
    names = list(models)
    fig, ax = plt.subplots(figsize=(SINGLE, 0.6 * SINGLE))
    width = 0.35
    for j, (variant, label, color) in enumerate((("eta@160", "$\\eta$, $\\mathrm{cc}(0.1)$", "#7f7f7f"),
                                                  ("xconf@160", "cross-fitted, $\\mathrm{xconf}(0.1)$", "#1f77b4"))):
        cells = [models[n]["cells"][f"{variant}|cc0.1"] for n in names]
        ax.bar([i + (j - 0.5) * width for i in range(len(names))], [c["speedup_warm"] for c in cells], width,
               yerr=[c["speedup_warm_sd"] for c in cells], capsize=3, color=color, label=label)
    ax.plot(range(len(names)), [models[n]["oracle"]["speedup_warm"] for n in names], "k_", ms=22, mew=2, label="oracle screen")
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set_xticks(range(len(names)), [f"NMF {n.split('|')[1]}\n$p = {models[n]['silent_rate']:.2f}$" for n in names])
    ax.set_ylabel("net speedup, warm fallback")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    save(fig, "layered_speedup.pdf")


def silent_failure() -> None:
    """One silent failure (membrane, NMF 24): displacement and multipliers along a row of candidate contact nodes."""
    import numpy as np

    d = np.load(RESULTS / "figure_silent_failure.npz")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(W_IN, 0.62 * W_IN), sharex=True,
                                      gridspec_kw={"height_ratios": [1.6, 1.0]})
    top.plot(d["x"], d["u_fom"], color="black", lw=2.6, alpha=0.45, label="full-order $u$")
    top.plot(d["x"], d["u_rom"], color="#1f77b4", lw=1.3, ls="--", label="reduced, lifted $\\tilde{u}$")
    top.plot(d["xc"], d["floor"], ls="none", marker="_", ms=9, mew=2, color="0.35", label="obstacle at contact nodes")
    xr = np.interp(d["xc"], d["x"], d["u_rom"])
    below = xr < d["floor"]
    top.plot(d["xc"][below], xr[below], ls="none", marker="o", mfc="none", mec="#d62728", ms=7, mew=1.4,
             label="penetration ($\\tilde{g} < 0$)")
    top.set_ylabel("displacement")
    top.grid(True, lw=0.4, alpha=0.5)
    bottom.plot(d["xc"], d["lam_fom"], ls="none", marker="o", mfc="none", mec="black", ms=6, label="full-order $\\lambda$")
    bottom.plot(d["xc"], d["lam_rom"], ls="none", marker="x", color="#1f77b4", ms=6, label="reduced, lifted $\\tilde{\\lambda}$")
    bottom.set_xlabel("$x$ along the row of candidate contact nodes")
    bottom.set_ylabel("contact multiplier")
    bottom.grid(True, lw=0.4, alpha=0.5)
    handles = top.get_legend_handles_labels()[0] + bottom.get_legend_handles_labels()[0]
    labels_ = top.get_legend_handles_labels()[1] + bottom.get_legend_handles_labels()[1]
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    save(fig, "silent_failure.pdf")


def roc_curves() -> None:
    """ROC curves of eta, the split-fitted and the cross-fitted learned indicator, averaged over replicates."""
    import numpy as np

    d, meta = np.load(RESULTS / "figure_roc.npz"), load("figure_roc.json")
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 0.5 * W_IN), sharey=True)
    styles = {"eta": ("$\\eta$", "#7f7f7f", "-"), "split": ("learned, split", "#ff7f0e", "--"),
              "xconf": ("learned, cross-fitted", "#1f77b4", "-")}
    for ax, (name, label) in zip(axes, (("block", "Block, NMF 8"), ("layered", "Layered block, NMF 12"))):
        for variant, (vlabel, color, ls) in styles.items():
            ax.plot(d["fpr_grid"], d[f"{name}|{variant}|tpr"], color=color, ls=ls, lw=1.6,
                    label=f"{vlabel} (AUC {meta[name]['auc_mean'][variant]:.2f})")
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
        ax.set_title(label)
        ax.set_xlabel("false-alarm rate")
        ax.legend(loc="lower right", frameon=False)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[0].set_ylabel("share of silent failures flagged")
    fig.tight_layout()
    save(fig, "roc_curves.pdf")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys
    sys.path.insert(0, str(ROOT))
    plt.rcParams.update(STYLE)
    for draw in (silent_failure, roc_curves, calibration_cost, outside_tradeoff, query_cost, layered_speedup):
        draw()
    if not math.isfinite(W_IN):
        raise ValueError("bad width")


if __name__ == "__main__":
    main()
