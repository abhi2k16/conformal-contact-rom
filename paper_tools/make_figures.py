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
    fig.savefig((FIGURES / name).with_suffix(".png"), dpi=220)          # for the Markdown export and quick viewing
    plt.close(fig)
    log.info("wrote %s", FIGURES / name)


def calibration_cost() -> None:
    """Left: calibration queries for cc(0.05) against p, every model, with n_min / p. Right: the same against the median energy error."""
    import sys
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from contactscreen.audit import load_audit
    from contactscreen.config import AUDIT_DIR
    from contactscreen.conformal import min_failures

    n_min = min_failures(0.05)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W_IN, 0.4 * W_IN))
    ps = [10 ** (x / 50) for x in range(-100, 1)]
    ax.plot(ps, [n_min / p for p in ps], color="0.5", lw=1.2, ls="--", label=f"$n_{{\min}}/p$, $n_{{\min}} = {n_min}$")
    for name, label, marker in PROBLEMS:
        rates = load(f"coverage_{name}.json")["rates"]
        x = [r["p"] for r in rates.values()]
        y = [r["queries_needed@0.05"] for r in rates.values()]
        xerr = [[r["p"] - r["p_ci"][0] for r in rates.values()], [r["p_ci"][1] - r["p"] for r in rates.values()]]
        ax.errorbar(x, y, xerr=xerr, fmt=marker, mfc="none", color=COLORS[name], ms=6, capsize=2, label=label)
        audits = [load_audit(f) for f in sorted(AUDIT_DIR.glob(f"{name}_seed*.npz"))]
        meds = {}
        for key in rates:
            kind, r = key.split("|")[:2]
            meds[key] = float(np.mean([np.median(a.arrays[f"pool|{kind}|{r}|err"]) for a in audits]))
        bx.plot([meds[k] for k in rates], y, marker, mfc="none", color=COLORS[name], ms=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("silent-failure rate $p$")
    ax.set_ylabel("calibration queries")
    ax.legend(loc="upper right", frameon=False)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.axvline(0.05, color="0.5", lw=1.0, ls=":")
    bx.set_xlabel("median relative energy error")
    for axis in (ax, bx):
        axis.grid(True, which="major", lw=0.4, alpha=0.5)
    ax.set_title("(a)", loc="left")
    bx.set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, "calibration_cost.pdf")


PAD_SHOWN = 0.05                                       # padded guard shown in the graded-shift figure
GRADED_MODEL = {"membrane": "nmf|48", "block": "nmf|8", "layered": "nmf|8"}      # most informative model of each benchmark


def graded_shift() -> None:
    """Miss rate and fallback share against the shift size, cc(0.1): eta, cross-fitted indicator, 5 % padded guard."""
    shifts = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 0.42 * W_IN))
    for name, label, marker in PROBLEMS:
        sh = load(f"shift_{name}.json")
        model = GRADED_MODEL[name]
        cell = lambda variant, guard, key, s: sh["cells"][f"{model}|{variant}|{guard}|cc0.1|{s}"][key]
        x = [100 * s for s in shifts]
        axes[0].plot(x, [cell("xconf", "none", "miss", s) for s in shifts], marker=marker, color=COLORS[name], ls="-",
                     mfc="none", label=f"{label}, NMF {model.split('|')[1]}")
        axes[0].plot(x, [cell("eta", "none", "miss", s) for s in shifts], marker=marker, color=COLORS[name], ls="--",
                     mfc="none", alpha=0.6)
        axes[1].plot(x, [cell("xconf", "none", "flagged", s) for s in shifts], marker=marker, color=COLORS[name], ls="-",
                     mfc="none")
        axes[1].plot(x, [cell("xconf", "pad0.05", "flagged", s) for s in shifts], color=COLORS[name], ls=":", lw=1.6)
    axes[0].axhline(0.1, color="black", lw=0.8)
    axes[0].text(20.4, 0.1, r" $\alpha$", va="center", fontsize=9)
    axes[0].set_ylabel("miss rate")
    axes[1].set_ylabel("fallback share")
    for ax in axes:
        ax.set_xlabel("shift outside the box (% of box width)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(True, lw=0.4, alpha=0.5)
    handles, labels_ = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", ls="-"), plt.Line2D([], [], color="0.3", ls="--", alpha=0.6),
                plt.Line2D([], [], color="0.3", ls=":", lw=1.6)]
    labels_ += ["cross-fitted indicator", r"analytical indicator $\eta$ (left only)",
                f"with {100 * PAD_SHOWN:.0f} % padded guard (right only)"]
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    save(fig, "graded_shift.pdf")


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


def fmt_p(x: float) -> str:
    """Rate format of the generated tables (paper_tools/result_numbers.fmt_rate)."""
    x = round(x, 12)
    return f"{x:.3f}" if abs(x) < 0.1 else f"{x:.2f}"


def layered_speedup() -> None:
    """Net speedup on the layered block, cc(0.1): eta against the cross-fitted indicator, warm fallback, with the exact-error screen."""
    models = load("learned_speedup_layered.json")["models"]
    names = list(models)
    fig, ax = plt.subplots(figsize=(SINGLE, 0.6 * SINGLE))
    width = 0.35
    for j, (variant, label, color) in enumerate((("eta@160", "$\\eta$, $\\mathrm{cc}(0.1)$", "#7f7f7f"),
                                                  ("xconf@160", "cross-fitted, $\\mathrm{xconf}(0.1)$", "#1f77b4"))):
        cells = [models[n]["cells"][f"{variant}|cc0.1"] for n in names]
        ax.bar([i + (j - 0.5) * width for i in range(len(names))], [c["speedup_warm"] for c in cells], width,
               yerr=[[c["speedup_warm"] - c["speedup_warm_ci"][0] for c in cells], [c["speedup_warm_ci"][1] - c["speedup_warm"] for c in cells]],
               capsize=3, color=color, label=label)
    ax.plot(range(len(names)), [models[n]["exact_error"]["speedup_warm"] for n in names], "k_", ms=22, mew=2, label="exact-error screen")
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set_xticks(range(len(names)), [f"NMF {n.split('|')[1]}\n$p = {fmt_p(models[n]['silent_rate'])}$" for n in names])
    ax.set_ylabel("net speedup, warm fallback")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    save(fig, "layered_speedup.pdf")


def _render_error_map(d):
    """Top view of the membrane mesh coloured by the pointwise displacement error |u_rom - u_fom| (fea_engine helpers).

    Candidate contact nodes: red spheres where the lifted displacement penetrates the obstacle, black where the full-order
    solution is in contact, small grey otherwise; the row of the profile panels is the dashed line. Returns (image, clim)."""
    import numpy as np
    import pyvista as pv

    from contactscreen.config import make_problem
    from fea_engine.visualization import to_pyvista

    problem = make_problem("membrane")
    grid = to_pyvista(problem.mesh)
    err = d["err_full"]
    grid.points[:, 2] = 0.0
    grid.point_data["err"] = err
    clim = (0.0, float(err.max()))
    pl = pv.Plotter(off_screen=True, window_size=[900, 900])
    pl.set_background("white")
    pl.add_mesh(grid, scalars="err", cmap="magma_r", clim=clim, show_edges=True, edge_color="#c0c0c0", line_width=0.3,
                show_scalar_bar=False, lighting=False)
    xy, pen, act = d["contact_xy"], d["contact_pen"], d["contact_active"]
    pts = np.column_stack([xy, np.full(len(xy), 0.02)])
    rest = ~(pen | act)
    pl.add_mesh(pv.PolyData(pts[rest]), color="#808080", render_points_as_spheres=True, point_size=5)
    pl.add_mesh(pv.PolyData(pts[act & ~pen]), color="black", render_points_as_spheres=True, point_size=11)
    pl.add_mesh(pv.PolyData(pts[pen]), color="#d62728", render_points_as_spheres=True, point_size=16)
    y = float(d["y_row"])
    pl.add_mesh(pv.Line((0, y, 0.03), (1, y, 0.03)), color="#1f77b4", line_width=3)
    pl.camera_position = "xy"
    pl.enable_parallel_projection()
    img = _crop_white(pl.screenshot(return_img=True))
    pl.close()
    return img, clim


def workflow() -> None:
    """Schematic of the offline calibration (panel a) and the online screened solve (panel b).

    Blue boxes are computations, orange hexagons decisions, green the accepted reduced answer, red the fallback solve.
    Coordinates are in units of 1 % of the figure width; every text is sized to fit inside its shape."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    blue, orange, green, red, grey = (("#eef3fa", "#1f77b4"), ("#fff3dc", "#d98c00"), ("#e6f4e7", "#2ca02c"),
                                      ("#fbe9e9", "#d62728"), ("#f2f2f2", "0.4"))
    fig, (a, b) = plt.subplots(2, 1, figsize=(W_IN, 0.45 * W_IN), gridspec_kw={"height_ratios": [22, 36], "hspace": 0.06})
    for ax, top in ((a, 22), (b, 36)):
        ax.set_xlim(-1, 101)
        ax.set_ylim(0, top)
        ax.set_axis_off()
    FS = 6.9

    def box(ax, x, y, w, h, text, colour=blue):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=colour[0], ec=colour[1], lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, linespacing=1.3)

    def hexagon(ax, x, y, w, h, text):
        k = 0.14 * w                                   # width of the pointed ends
        ax.add_patch(Polygon([(x, y + h / 2), (x + k, y + h), (x + w - k, y + h), (x + w, y + h / 2), (x + w - k, y), (x + k, y)],
                             closed=True, fc=orange[0], ec=orange[1], lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, linespacing=1.3)

    def arrow(ax, p0, p1, label=None, label_at=None):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.1, color="0.25", shrinkA=0, shrinkB=0))
        if label:
            x, y = label_at if label_at else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 1.6)
            ax.text(x, y, label, ha="center", va="bottom", fontsize=6.6, color="0.15")

    # (a) offline
    a.text(-1, 21.8, "(a) Offline, once", fontsize=9, fontweight="bold", va="top")
    w, gap, y, h = 22.0, 4.0, 1.0, 16.0
    xs = [i * (w + gap) for i in range(4)]
    box(a, xs[0], y, w, h, "training queries:\nfull-order solves,\nPOD basis $\\mathbf{V}$,\ndual basis $\\mathbf{U}$")
    box(a, xs[1], y, w, h, "$N_{\\mathrm{cal}}$ calibration\nqueries: full-order\nand reduced solve,\nerror $e$, label,\nfeatures")
    box(a, xs[2], y, w, h, "$K$ fold indicators\n$s^{(j)}$; out-of-fold\nscores $o_i$ of $n$\nfailures; rank $k$")
    box(a, xs[3], y, w, h, "stored: $\\mathbf{V}$, $\\mathbf{U}$,\n$s^{(j)}$, $o_i$, $k$,\ntraining box", colour=grey)
    for i in range(3):
        arrow(a, (xs[i] + w + 0.5, y + h / 2), (xs[i + 1] - 0.5, y + h / 2))

    # (b) online: hexagons are decisions
    b.text(-1, 35.8, "(b) Online, for every query $\\mu$", fontsize=9, fontweight="bold", va="top")
    d, w, ymid, hb, hd = 14.0, 12.0, 24.5, 11.5, 11.5
    gap = (100 - (3 * d + 3 * w)) / 5
    xpos, cur = [], 0.0
    for width in (d, w, d, w, d, w):
        xpos.append(cur)
        cur += width + gap
    widths = (d, w, d, w, d, w)
    hexagon(b, xpos[0], ymid - hd / 2, d, hd, "$\\mu$ in\nthe box?")
    box(b, xpos[1], ymid - hb / 2, w, hb, "reduced\nsolve, lift\n($t_{\\mathrm{R}}$)")
    hexagon(b, xpos[2], ymid - hd / 2, d, hd, "solver\nconverged?")
    box(b, xpos[3], ymid - hb / 2, w, hb, "features,\nscores, $c$\n($t_{\\mathrm{ind}}$)")
    hexagon(b, xpos[4], ymid - hd / 2, d, hd, "$c < k$?")
    box(b, xpos[5], ymid - hb / 2, w, hb, "accept\nreduced\nanswer", colour=green)
    for i in range(5):
        arrow(b, (xpos[i] + widths[i] + 0.3, ymid), (xpos[i + 1] - 0.3, ymid), "yes" if i in (0, 2, 4) else None)
    # fallback box below the decisions; each "no" branch drops straight onto it
    fb_x0, fb_x1, fb_y, fb_h = xpos[0], xpos[4] + d, 0.8, 7.5
    box(b, fb_x0, fb_y, fb_x1 - fb_x0, fb_h, "fallback solve: full-order model, warm-started\nfrom the reduced active set ($t_{\\mathrm{FO}}$)", colour=red)
    for i in (0, 2, 4):
        cx = xpos[i] + d / 2
        arrow(b, (cx, ymid - hd / 2 - 0.3), (cx, fb_y + fb_h + 0.3), "no", label_at=(cx + 2.2, (ymid - hd / 2 + fb_y + fb_h) / 2 - 0.8))
    save(fig, "workflow.pdf")


def silent_failure() -> None:
    """One silent failure (membrane, NMF 24): profiles along a row of candidate contact nodes and the 2-D error map."""
    import numpy as np
    from matplotlib import cm, colors

    d = np.load(RESULTS / "figure_silent_failure.npz")
    blue, red = "#1f77b4", "#d62728"
    fig = plt.figure(figsize=(W_IN, 0.66 * W_IN))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1.0], height_ratios=[1.3, 1.0, 1.0], hspace=0.42, wspace=0.3)
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])
    for a in (ax1, ax2):
        a.tick_params(labelbottom=False)
    # (a) displacement at the candidate contact nodes against the obstacle
    ax1.plot(d["x"], d["u_fom"], color="black", lw=2.4, alpha=0.4, label="full-order $u$")
    ax1.plot(d["x"], d["u_rom"], color=blue, lw=1.2, ls="--", label="reduced, lifted $\\tilde{u}$")
    ax1.plot(d["xc"], d["floor"], ls="none", marker="_", ms=9, mew=2, color="0.35", label="obstacle")
    ax1.set_ylabel("displacement")
    # (b) gap: full-order and lifted; negative lifted gap is penetration
    ax2.axhline(0.0, color="0.3", lw=0.8)
    ax2.fill_between(d["xc"], np.minimum(d["gap_rom"], 0.0), 0.0, color=red, alpha=0.45, lw=0)
    ax2.plot(d["xc"], d["gap_fom"], ls="none", marker="o", mfc="none", mec="black", ms=5.5, label="full-order $g$")
    ax2.plot(d["xc"], d["gap_rom"], ls="none", marker="x", color=blue, ms=5.5, label="reduced, lifted $\\tilde{g}$")
    pen = d["gap_rom"] < 0
    ax2.plot(d["xc"][pen], d["gap_rom"][pen], ls="none", marker="o", mfc="none", mec=red, ms=8, mew=1.3,
             label="penetration ($\\tilde{g} < 0$)")
    ax2.set_yscale("symlog", linthresh=0.01)
    ax2.set_yticks([-0.01, 0, 0.01, 0.1, 0.5])
    ax2.set_yticklabels(["-0.01", "0", "0.01", "0.1", "0.5"])
    ax2.set_ylabel("gap (symlog)")
    # (c) multipliers
    ax3.plot(d["xc"], d["lam_fom"], ls="none", marker="o", mfc="none", mec="black", ms=5.5, label="full-order $\\lambda$")
    ax3.plot(d["xc"], d["lam_rom"], ls="none", marker="x", color=blue, ms=5.5, label="reduced, lifted $\\tilde{\\lambda}$")
    ax3.set_ylabel("multiplier")
    ax3.set_xlabel("$x$ along the row of candidate contact nodes")
    for a, tag in ((ax1, "(a)"), (ax2, "(b)"), (ax3, "(c)")):
        a.grid(True, lw=0.4, alpha=0.5)
        a.text(0.0, 1.03, tag, transform=a.transAxes, va="bottom", ha="left", fontsize=9, fontweight="bold")
    # (d) pointwise error over the membrane, rendered with fea_engine / PyVista
    axm = fig.add_subplot(gs[:, 1])
    img, clim = _render_error_map(d)
    axm.imshow(img)
    axm.set_axis_off()
    axm.set_title("(d) displacement error $|\\tilde{u} - u|$", fontsize=9)
    sm = cm.ScalarMappable(norm=colors.Normalize(*clim), cmap="magma_r")
    cb = fig.colorbar(sm, ax=axm, orientation="horizontal", fraction=0.05, pad=0.03, aspect=18)
    cb.ax.tick_params(labelsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="black", lw=2.4, alpha=0.4), Line2D([], [], ls="none", marker="o", mfc="none", mec="black", ms=6),
               Line2D([], [], color=blue, lw=1.2, ls="--"), Line2D([], [], ls="none", marker="x", color=blue, ms=6),
               Line2D([], [], ls="none", marker="_", mew=2, color="0.35", ms=9),
               Line2D([], [], ls="none", marker="o", mfc="none", mec=red, mew=1.3, ms=8), Line2D([], [], ls="none", marker="o", color=red, ms=6),
               Line2D([], [], ls="none", marker="o", color="black", ms=5), Line2D([], [], color=blue, lw=2.5)]
    labels_ = ["full-order (line)", "full-order (nodes)", "reduced, lifted (line)", "reduced, lifted (nodes)", "obstacle",
               "penetration in (b)", "penetrating node in (d)", "full-order contact in (d)", "profile row in (d)"]
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, fontsize=8, columnspacing=1.5)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.27)
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


def _sketch(ax, kind: str, meta: dict) -> None:
    """Side view (x-z, through the load or foundation centre) of one benchmark; not to scale."""
    import numpy as np
    from matplotlib.patches import Rectangle

    x = np.linspace(0.0, 1.0, 101)
    gap_draw = 0.12                                   # drawn gap at the centre (exaggerated)
    if kind == "membrane":
        ax.plot([0, 1], [0, 0], color="black", lw=2.2)
        floor = -(gap_draw + 0.35 * (x - 0.5) ** 2 / 0.25)
        for xs in (0.0, 1.0):
            ax.plot([xs], [0], marker="^", color="black", ms=8)
        top, body_label = 0.0, None
    else:
        depth = 0.5
        top = depth
        if meta["layered"]:
            ax.add_patch(Rectangle((0, 0), 1, depth / 2, facecolor="#c6dbef", edgecolor="black", lw=1.0))
            ax.add_patch(Rectangle((0, depth / 2), 1, depth / 2, facecolor="#fdd0a2", edgecolor="black", lw=1.0))
            ax.text(0.5, 0.75 * depth, "$\\rho_E E$", ha="center", va="center", fontsize=9)
            ax.text(0.5, 0.25 * depth, "$E$", ha="center", va="center", fontsize=9)
        else:
            ax.add_patch(Rectangle((0, 0), 1, depth, facecolor="#e0e0e0", edgecolor="black", lw=1.0))
        for xs, sign in ((0.0, -1), (1.0, 1)):
            for z in np.linspace(0.02, depth - 0.02, 6):
                ax.plot([xs, xs + 0.05 * sign], [z, z + 0.05], color="black", lw=0.8)
        floor = -(gap_draw + 0.35 * (x - 0.5) ** 2 / 0.25)
        body_label = None
    for xa in np.linspace(0.1, 0.9, 9):              # Gaussian load or pressure, downward
        h = 0.32 * np.exp(-(xa - 0.5) ** 2 / (2 * 0.18 ** 2))
        ax.annotate("", xy=(xa, top + 0.02), xytext=(xa, top + 0.02 + h),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#d62728"))
    ax.plot(x, floor, color="0.25", lw=1.2)
    ax.fill_between(x, floor, floor.min() - 0.15, color="0.85", hatch="///", edgecolor="0.6", lw=0)
    ax.annotate("", xy=(0.5, floor[50]), xytext=(0.5, 0.0), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(0.53, floor[50] / 2, "gap", fontsize=8, va="center")
    if body_label:
        ax.text(0.02, 0.06, body_label, fontsize=8)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(floor.min() - 0.15, top + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")


RENDER_PX = (1100, 900)                         # off-screen render size (pixels) of one benchmark panel
FIELD_CMAP = "viridis"


def _crop_white(img):
    """Remove white margins from an RGB(A) render."""
    import numpy as np

    rgb = img[..., :3].astype(int)
    keep = np.where((rgb < 250).any(axis=2))
    return img[keep[0].min():keep[0].max() + 1, keep[1].min():keep[1].max() + 1]


def _render_benchmark(name: str, d, meta: dict):
    """Full-order solution of one benchmark at true scale, drawn with fea_engine's PyVista helpers.

    Membrane: the 2-D mesh lifted by the out-of-plane displacement, over the floor surface. Blocks: the deformed hex
    mesh (``fea_engine.visualization.deformed_grid``) seen from below, over the foundation surface. Candidate contact
    nodes in contact are red spheres. Returns (image array, colour limits, field label)."""
    import numpy as np
    import pyvista as pv

    from contactscreen.config import make_problem
    from fea_engine.visualization import deformed_grid, to_pyvista

    problem = make_problem(name)
    U, mu = d[f"{name}|U_full"], d[f"{name}|mu"]
    nodes_c, active = d[f"{name}|contact_nodes"], d[f"{name}|active"] > 0.5
    pl = pv.Plotter(off_screen=True, window_size=list(RENDER_PX))
    pl.set_background("white")
    if name == "membrane":
        grid = to_pyvista(problem.mesh)
        grid.points[:, 2] = U                               # out-of-plane displacement as height
        field, label = U, "displacement $u$"
        cx, cy, d0, kappa = 0.5, 0.5, meta["d0"], meta["kappa"]
    else:
        grid = deformed_grid(problem.mesh, U, scale=1.0)   # true scale
        field, label = U[2::3], "displacement $u_z$"
        p0 = 1 if meta["layered"] else 0
        cx, cy, d0, kappa = mu[p0 + 1], mu[p0 + 2], meta["d0"], meta["kappa"]
    clim = (float(field.min()), float(field.max()))
    grid.point_data["field"] = field
    pl.add_mesh(grid, scalars="field", cmap=FIELD_CMAP, clim=clim, show_edges=True, edge_color="#404040",
                line_width=0.4, show_scalar_bar=False, ambient=0.55, diffuse=0.5)   # faces seen from below stay readable
    xs = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(xs, xs)
    Z = -(d0 + kappa * ((X - cx) ** 2 + (Y - cy) ** 2))
    floor = pv.StructuredGrid(X, Y, Z)
    pl.add_mesh(floor, style="wireframe", color="#7f7f7f", opacity=0.6, line_width=1.0)   # obstacle: see-through
    pts = grid.points[nodes_c]
    if active.any():
        pl.add_mesh(pv.PolyData(pts[active]), color="#d62728", render_points_as_spheres=True, point_size=24)
    if (~active).any():
        pl.add_mesh(pv.PolyData(pts[~active]), color="#202020", render_points_as_spheres=True, point_size=8)
    if name == "membrane":
        pl.camera_position = [(1.7, -1.3, 1.5), (0.5, 0.5, -0.3), (0, 0, 1)]
    else:
        pl.camera_position = [(1.5, -1.1, -1.6), (0.5, 0.5, 0.05), (0, 0, 1)]  # from below: the contact face
    img = _crop_white(pl.screenshot(return_img=True))
    pl.close()
    return img, clim, label


def benchmarks() -> None:
    """Benchmark sketches (top) and one full-order solution in contact per benchmark, rendered with fea_engine (bottom)."""
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    d, meta = np.load(RESULTS / "figure_benchmarks.npz"), load("figure_benchmarks.json")
    fig, axes = plt.subplots(2, 3, figsize=(W_IN, 0.7 * W_IN), gridspec_kw={"height_ratios": [0.75, 1.0]})
    titles = {"membrane": "Membrane", "block": "Block", "layered": "Layered block"}
    for j, name in enumerate(("membrane", "block", "layered")):
        _sketch(axes[0, j], name, meta[name])
        axes[0, j].set_title(titles[name])
        ax = axes[1, j]
        img, clim, label = _render_benchmark(name, d, meta[name])
        ax.imshow(img)
        ax.axis("off")
        sm = ScalarMappable(norm=Normalize(*clim), cmap=FIELD_CMAP)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.04, aspect=18)
        cb.set_ticks([clim[0], 0.5 * (clim[0] + clim[1]), clim[1]])
        cb.set_ticklabels([f"{v:.2g}" for v in (clim[0], 0.5 * (clim[0] + clim[1]), clim[1])])
        cb.ax.tick_params(labelsize=8)
        cb.set_label(label, fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="none", color="#d62728", ms=5, label="contact node, in contact"),
               plt.Line2D([], [], marker="o", ls="none", color="#202020", ms=3, label="not in contact"),
               plt.Line2D([], [], ls="-", color="#7f7f7f", lw=1.0, label="rigid obstacle")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "benchmarks.pdf")


def threshold_figure() -> None:
    """Indicator values of calibration failures and good answers, and the class-conditional threshold."""
    import numpy as np

    d, meta = np.load(RESULTS / "figure_threshold.npz"), load("figure_threshold.json")
    rng = np.random.default_rng(0)                    # jitter only
    fig, axes = plt.subplots(2, 1, figsize=(W_IN, 0.55 * W_IN))
    titles = {"membrane": "Membrane, NMF 24", "block": "Block, NMF 12"}
    for ax, name in zip(axes, ("membrane", "block")):
        m = meta[name]
        fail, good = d[f"{name}|fail"], d[f"{name}|good"]
        floor = 0.5 * min(fail[fail > 0].min(), good[good > 0].min())
        fail, good = np.maximum(fail, floor), np.maximum(good, floor)
        lo, hi = min(fail.min(), good.min()) / 1.5, max(fail.max(), good.max()) * 1.5
        ax.axvspan(lo, m["tau"], color="#e5f5e0", lw=0)
        ax.axvspan(m["tau"], hi, color="#fee0d2", lw=0)
        ax.plot(fail, 1 + 0.12 * rng.uniform(-1, 1, len(fail)), "x", color="#d62728", ms=5, label="silent failures")
        ax.plot(good, 0 + 0.12 * rng.uniform(-1, 1, len(good)), "o", mfc="none", mec="#1f77b4", ms=4, label="good answers")
        ax.axvline(m["tau"], color="black", lw=1.2)
        ax.text(m["tau"], 1.45, f"  $\\tau$: $k$-th smallest failure score, $k = {m['k']}$", fontsize=8, va="bottom")
        ax.text(lo * 1.1, 0.5, f"accepted\n{m['n_fail_accepted']} of {m['n_fail']} failures", fontsize=8, va="center")
        ax.text(hi / 1.1, 0.5, f"sent to fallback\n{m['n_good_flagged']} of {m['n_good']} good answers", fontsize=8,
                va="center", ha="right")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.5, 1.9)
        ax.set_yticks([0, 1], ["good", "failure"])
        ax.set_title(titles[name], fontsize=9.5, loc="left")
    axes[-1].set_xlabel("analytical indicator $\\eta$ (logarithmic)")
    fig.tight_layout()
    save(fig, "threshold.pdf")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys
    sys.path.insert(0, str(ROOT))
    plt.rcParams.update(STYLE)
    for draw in (workflow, silent_failure, roc_curves, benchmarks, threshold_figure, calibration_cost, outside_tradeoff, graded_shift,
                 query_cost, layered_speedup):
        draw()
    if not math.isfinite(W_IN):
        raise ValueError("bad width")


if __name__ == "__main__":
    main()
