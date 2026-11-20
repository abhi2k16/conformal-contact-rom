"""Result numbers for the manuscript: every value of ``results/*.json`` as a key, plus derived summaries.

Flattening: the file stem (``_`` -> ``.``) and the JSON path give the key; ``|`` in experiment cell names becomes ``.``,
the segment ``cells`` is dropped, and the two ends of an interval list become ``.0`` and ``.1``. Example:
``crossfit_block.json`` -> ``cells`` -> ``nmf|8|240|xconf|cc0.1`` -> ``fpr`` gives ``crossfit.block.nmf.8.240.xconf.cc0.1.fpr``.
The format follows the field name (speedups one decimal, times in ms, counts as integers, rates two or three decimals),
so the text never rounds by hand. Derived summaries (maxima and ranges over models) are named ``sum.*``.
Used by ``make_numbers.py``.
"""

import json
import math
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
SKIP = {"environment", "seeds"}
COUNT_FIELDS = ("n_per_shift", "n_active", "k", "n_fail", "n_good", "n_fail_accepted", "n_good_flagged", "active_fom", "n_contact", "penetrating_nodes", "pool_index", "n_silent_test_per_replicate",
                "n_good_test_per_replicate", "n_orders_M", "n_silent", "missed", "n_eval", "n_time", "draws", "n_replicates", "n_pool", "n_splits", "n_boot",
                "n_cal", "repeats", "replicate", "budget", "n_draws", "solves_with_fallback_route", "queries_needed")
EPS_ALPHA = 1e-12
S_TOL = 0.01                  # experiment S: change of the false-alarm rate counted as a difference


def fmt_rate(x: float) -> str:
    return f"{x:.3f}" if abs(x) < 0.1 else f"{x:.2f}"


def fmt_ms(x: float) -> str:
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 1 else f"{x:.2f}")


def fmt_sci(x: float) -> str:
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    return f"{x / 10 ** e:.0f}\\times 10^{{{e}}}"


def fmt_miss(x: float) -> str:
    """Miss rates and risks are compared with alpha and delta, so always three decimals."""
    return f"{x:.3f}"


def fmt_value(path: str, x: float) -> str:
    name = re.split(r"\.(?=[A-Za-z_])", path)[-1]          # field name; segments such as "@0.05" contain dots
    if "miss" in path.lower() and "missed" not in name or name in ("risk", "risk_ci") or ".risk_ci." in path:
        return fmt_miss(x)
    if ".M_sizes." in path and name in ("mean", "p90"):                     # calibration sizes of experiment M
        return f"{round(x):,}".replace(",", "\\,")
    if name in COUNT_FIELDS or name.startswith(("queries_needed", "need@")):          # match the field name, not a substring
        return f"{x:g}" if x != round(x) else f"{int(round(x)):,}".replace(",", "\\,")
    if name in ("budgets", "lams", "margins", "fit_sizes", "fit_shares", "budget"):     # design values: as written in the scripts
        return f"{x:g}"
    if name == "defined":
        return f"{x:.2f}"
    if "speedup" in name or "break_even" in name:
        return f"{x:.1f}"
    if name.endswith("_ms") or "_ms_" in name or name == "setup_ms":
        return fmt_ms(x)
    if name.startswith("it_") or "iterations" in name:
        return f"{x:g}"
    if name.startswith("max_du") or name.startswith("max_rel"):
        return fmt_sci(x)
    if name in ("rel", "chosen_rel"):
        return f"{x:g}"
    if "failures_in_calibration" in name or "ess_" in name:
        return f"{x:.1f}"
    return fmt_rate(x)


def flatten(obj, prefix: str, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP:
                continue
            seg = str(k).replace("|", ".")
            flatten(v, prefix if k == "cells" else f"{prefix}.{seg}", out)
    elif isinstance(obj, list):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            for i, v in enumerate(obj):
                out[f"{prefix}.{i}"] = fmt_value(prefix, float(v))
        elif all(isinstance(v, dict) for v in obj):          # records, e.g. warm-start rows per seed threshold
            for i, v in enumerate(obj):
                flatten(v, f"{prefix}.{i}", out)
    elif isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    elif isinstance(obj, (int, float)):
        out[prefix] = fmt_value(prefix, float(obj))


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def flat_numbers() -> dict:
    out: dict = {}
    for path in sorted(RESULTS.glob("*.json")):
        flatten(json.loads(path.read_text(encoding="utf-8")), path.stem.replace("_", "."), out)
    return out


# --------------------------------------------------------------------------------------------------- summaries

PROBLEMS = ("membrane", "block", "layered")
NCAL_A = {"membrane": 160, "block": 160, "layered": 80}          # calibration size quoted for experiment A


def nmf_models(cov: dict) -> list:
    return [k for k in cov["rates"] if k.startswith("nmf")]


def summary_numbers() -> dict:
    out = {}
    # A: rules at the quoted calibration size
    risks = {"crc0.01": [], "crc0.02": []}
    worst_heur = (-1.0, None, None)
    for name in PROBLEMS:
        cov, n = load(f"coverage_{name}.json"), NCAL_A[name]
        out[f"sum.A.{name}.ncal"] = str(n)
        fprs = []
        for m in nmf_models(cov):
            cell = lambda rule: cov["cells"][f"{m}|{n}|{rule}"]
            if cell("cc0.1")["fpr"] is not None:
                fprs.append(cell("cc0.1")["fpr"])
            for rule in risks:
                if cell(rule)["defined"] > 0:
                    risks[rule].append(cell(rule)["risk"])
            h = cell("heur")
            if h["miss"] is not None and h["miss"] > worst_heur[0]:
                worst_heur = (h["miss"], f"{name}.{m}", h["miss_ci"])
        out[f"sum.A.{name}.cc0.1.fpr.min"], out[f"sum.A.{name}.cc0.1.fpr.max"] = fmt_rate(min(fprs)), fmt_rate(max(fprs))
    for rule, vals in risks.items():
        out[f"sum.A.{rule}.risk.min"], out[f"sum.A.{rule}.risk.max"] = fmt_miss(min(vals)), fmt_miss(max(vals))
    out["sum.A.heur.maxmiss"] = fmt_miss(worst_heur[0])
    out["sum.A.heur.maxmiss.lo"], out["sum.A.heur.maxmiss.hi"] = fmt_miss(worst_heur[2][0]), fmt_miss(worst_heur[2][1])
    for alpha in ("0.05", "0.1"):
        worst = max(load(f"coverage_{p}.json")["cells"][f"{m}|{NCAL_A[p]}|cc{alpha}"]["miss"] or 0.0
                    for p in PROBLEMS for m in load(f"coverage_{p}.json")["rates"])
        out[f"sum.A.cc{alpha}.maxmiss"] = fmt_miss(worst)
    above = 0
    for p in PROBLEMS:
        for k, c in load(f"coverage_{p}.json")["cells"].items():
            rule = k.split("|")[3]
            if rule.startswith("cc") and c["miss_ci"] and c["miss_ci"][0] > float(rule[2:]) + EPS_ALPHA:
                above += 1
    out["sum.A.cc.ciabovealpha"] = str(above)          # cells (all models, sizes) whose interval lies above alpha
    # K: cross-fitted rule, all cells
    for alpha in ("0.05", "0.1"):
        misses = [c["miss"] for p in PROBLEMS for k, c in load(f"crossfit_{p}.json")["cells"].items()
                  if f"|xconf|cc{alpha}" in k and c["miss"] is not None]
        out[f"sum.K.xconf.cc{alpha}.maxmiss"] = fmt_miss(max(misses))
        out[f"sum.K.xconf.cc{alpha}.ciabove"] = str(sum(1 for pp in PROBLEMS for k, c in load(f"crossfit_{pp}.json")["cells"].items()
                                                         if f"|xconf|cc{alpha}" in k and c["miss_ci"] and c["miss_ci"][1] > float(alpha)))
        out[f"sum.K.xconf.cc{alpha}.cells"] = str(len(misses))
    # B: calibration size for the most accurate model of each benchmark
    for name in PROBLEMS:
        rates = load(f"coverage_{name}.json")["rates"]
        best = min(nmf_models({"rates": rates}), key=lambda m: rates[m]["p"])
        out[f"sum.B.{name}.best"] = best.split("|")[1]
    # G: AUC of indicator v1 and v2 per benchmark (ranges over NMF models)
    for name in PROBLEMS:
        for tag, f in (("v1", f"coverage_{name}.json"), ("v2", f"coverage_{name}_eta2.json")):
            aucs = [r["auc"] for k, r in load(f)["rates"].items() if k.startswith("nmf") and r["auc"] is not None]
            out[f"sum.G.{name}.{tag}.auc.min"], out[f"sum.G.{name}.{tag}.auc.max"] = fmt_rate(min(aucs)), fmt_rate(max(aucs))
    # J: speedups on layered, ranges over models
    sp = load("learned_speedup_layered.json")["models"]
    for variant in ("eta@160", "xconf@160"):
        for kind in ("speedup_cold", "speedup_warm"):
            vals = [v["cells"][f"{variant}|cc0.1"][kind] for v in sp.values()]
            out[f"sum.J.{variant.split('@')[0]}.cc0.1.{kind}.min"] = f"{min(vals):.1f}"
            out[f"sum.J.{variant.split('@')[0]}.cc0.1.{kind}.max"] = f"{max(vals):.1f}"
    flag_all = [v["cells"]["eta@160|flag_all"]["speedup_warm"] for v in sp.values()]
    out["sum.J.flagall.warm.min"], out["sum.J.flagall.warm.max"] = f"{min(flag_all):.1f}", f"{max(flag_all):.1f}"
    oracle = [v["oracle"]["speedup_cold"] for v in sp.values()]
    out["sum.J.oracle.cold.min"], out["sum.J.oracle.cold.max"] = f"{min(oracle):.1f}", f"{max(oracle):.1f}"
    xmiss = [v["cells"]["xconf@160|cc0.1"]["miss_rate"] for v in sp.values()]
    out["sum.J.xconf.cc0.1.miss.max"] = fmt_miss(max(xmiss))
    # H: condensed solver against the suite solver
    for name in ("membrane", "block"):
        h = load(f"condensed_{name}.json")["H"]
        valid = [r for m in h["models"].values() for rule, r in m["rules"].items() if rule in ("cc0.05", "cc0.1")]
        out[f"sum.H.{name}.valid.cold.max"] = f"{max(r['speedup_cold'] for r in valid):.2f}"
        out[f"sum.H.{name}.valid.warm.max"] = f"{max(r['speedup_warm'] for r in valid):.2f}"
        out[f"sum.H.{name}.oracle.cold.max"] = f"{max(m['rules']['oracle']['speedup_cold'] for m in h['models'].values()):.2f}"
        out[f"sum.H.{name}.valid.cold.min"] = f"{min(r['speedup_cold'] for r in valid):.2f}"
    # J: one-time costs per replicate (means over replicates) and break-even with all of them
    j = load("learned_speedup_layered.json")
    one = j["one_time"]
    out["sum.J.calib.seconds"] = f"{one['t_calibration_solves_s']:.0f}"
    out["sum.J.build.seconds"] = f"{one['t_build_s']:.0f}"
    out["sum.J.ntrain"] = str(one["n_train"])
    fits = [v["t_fit_xconf_s"] for v in sp.values()]
    out["sum.J.fit.min"], out["sum.J.fit.max"] = f"{min(fits):.1f}", f"{max(fits):.1f}"
    for r, v in sp.items():
        n = r.split("|")[1]
        c = v["cells"]["xconf@160|cc0.1"]
        out[f"sum.J.cost.{n}.fit"] = f"{v['t_fit_xconf_s']:.1f}"
        out[f"sum.J.cost.{n}.onetime"] = f"{one['t_calibration_solves_s'] + v['t_fit_xconf_s']:.0f}"
        out[f"sum.J.cost.{n}.onetime_all"] = f"{one['t_build_s'] + one['t_calibration_solves_s'] + v['t_fit_xconf_s']:.0f}"
        out[f"sum.J.cost.{n}.rom"], out[f"sum.J.cost.{n}.feat"] = f"{v['t_rom_ms_mean']:.1f}", f"{v['t_feat_ms_mean']:.1f}"
        out[f"sum.J.cost.{n}.cold"], out[f"sum.J.cost.{n}.warm"] = f"{v['t_cold_ms_mean']:.0f}", f"{v['t_warm_ms_mean']:.0f}"
        for kind in ("cold", "warm"):
            s = c[f"speedup_{kind}"]
            pipe = v["t_cold_ms_mean"] / s                                     # mean pipeline time per query, ms
            out[f"sum.J.cost.{n}.pipe_{kind}"] = f"{pipe:.0f}"
            if s > 1.0:
                saving = v["t_cold_ms_mean"] - pipe                            # ms saved per query against the cold full-order solve
                out[f"sum.J.cost.{n}.q_{kind}"] = f"{1e3 * (one['t_calibration_solves_s'] + v['t_fit_xconf_s']) / saving:.0f}"
                out[f"sum.J.cost.{n}.qall_{kind}"] = f"{1e3 * (one['t_build_s'] + one['t_calibration_solves_s'] + v['t_fit_xconf_s']) / saving:.0f}"
            else:
                out[f"sum.J.cost.{n}.q_{kind}"] = out[f"sum.J.cost.{n}.qall_{kind}"] = "--"
    nsil = [n for v in sp.values() for n in v["n_silent_per_replicate"]]
    out["sum.J.nsilent.min"], out["sum.J.nsilent.max"] = str(min(nsil)), str(max(nsil))
    for tag in ("q", "qall"):
        vals = [float(out[f"sum.J.cost.{r.split('|')[1]}.{tag}_warm"]) for r in sp if out[f"sum.J.cost.{r.split('|')[1]}.{tag}_warm"] != "--"]
        out[f"sum.J.{tag}warm.min"], out[f"sum.J.{tag}warm.max"] = f"{min(vals):.0f}", f"{max(vals):.0f}"
    # Break-even number of online queries: the budget of calibration solves (each about one cold full-order solve)
    # is repaid when the saving per query, t_cold (1 - 1/S), has accumulated: Q = budget / (1 - 1/S), S = net speedup.
    for kind in ("speedup_cold", "speedup_warm"):
        q = [j["budget"] / (1.0 - 1.0 / v["cells"]["xconf@160|cc0.1"][kind]) for v in sp.values()
             if v["cells"]["xconf@160|cc0.1"][kind] > 1.0]
        out[f"sum.J.breakeven.{kind}.min"], out[f"sum.J.breakeven.{kind}.max"] = f"{min(q):.0f}", f"{max(q):.0f}"
    # L: cells (variant, eps', alpha, model, benchmark) in which lower-threshold calibration beat the standard rule,
    # i.e. was defined more often or had a lower false-alarm rate by more than TOL; M: largest |miss - alpha| where defined
    tol, total, dev = 0.005, 0, {0.05: 0.0, 0.1: 0.0}
    fpr_lower, cp_def_more, oracle_def_more, oracle_def_gain = 0, 0, 0, 0.0
    for name in PROBLEMS:
        ff = load(f"fewfailures_{name}.json")
        for k, c in ff["L"].items():
            kind, r, variant, alpha, eps_low = k.split("|")
            if variant == "standard":
                continue
            std = ff["L"][f"{kind}|{r}|standard|{alpha}|{ff['eps']}"]
            total += 1
            fpr_lower += int(c["fpr"] is not None and std["fpr"] is not None and c["fpr"] < std["fpr"] - tol)
            if c["defined"] > std["defined"] + tol:
                if variant == "cp":
                    cp_def_more += 1
                else:
                    oracle_def_more += 1
                    oracle_def_gain = max(oracle_def_gain, c["defined"] - std["defined"])
        for k, c in ff["M"].items():
            alpha = float(k.split("|")[3])
            if c["defined"] == 1.0 and c["miss"] is not None:
                dev[alpha] = max(dev[alpha], abs(c["miss"] - alpha))
    out.update({"sum.L.cells": str(total), "sum.L.fprlower": str(fpr_lower), "sum.L.cp.defmore": str(cp_def_more),
                "sum.L.oracle.defmore": str(oracle_def_more), "sum.L.oracle.defgain": f"{oracle_def_gain:.2f}"})
    out["sum.M.maxdev.0.05"], out["sum.M.maxdev.0.1"] = fmt_miss(dev[0.05]), fmt_miss(dev[0.1])
    # S: residual-projection features against the base learned indicator, every cell (model, budget, alpha, variant)
    lower, higher, cells, best_drop, worst_rise, s_miss = 0, 0, 0, 0.0, 0.0, 0.0
    for name in PROBLEMS:
        rp = load(f"resproj_{name}.json")
        for k, c in rp["cells"].items():
            model, r, budget, variant, rule = k.split("|")
            if variant in ("eta", "base") or c["fpr"] is None:
                continue
            base = rp["cells"][f"{model}|{r}|{budget}|base|{rule}"]
            if base["fpr"] is None:
                continue
            cells += 1
            delta = c["fpr"] - base["fpr"]
            lower += int(delta < -S_TOL)
            higher += int(delta > S_TOL)
            best_drop, worst_rise = min(best_drop, delta), max(worst_rise, delta)
            if rule == "cc0.1" and c["miss"] is not None:
                s_miss = max(s_miss, c["miss"])
    out.update({"sum.S.tol": f"{S_TOL:g}", "sum.S.cells": str(cells), "sum.S.lower": str(lower), "sum.S.higher": str(higher),
                "sum.S.bestdrop": f"{-best_drop:.3f}", "sum.S.worstrise": f"{worst_rise:.3f}", "sum.S.maxmiss.cc0.1": fmt_miss(s_miss)})
    # T: deterministic residual bound
    viol, comp_max = 0, {}
    for name in PROBLEMS:
        b = load(f"bound_{name}.json")["models"]
        viol += sum(v["violations"] for v in b.values())
        comp_max[name] = max(v["computable"] for v in b.values())
        out[f"sum.T.{name}.computable.max"] = fmt_rate(comp_max[name])
        out[f"sum.T.{name}.fallback.min"] = f"{min(v['screen']['flagged'] for v in b.values()):.3f}"
        pg = [v["penetration_good_median"] for v in b.values()]
        ps = [v["penetration_silent_median"] for v in b.values()]
        out[f"sum.T.{name}.pengood.min"], out[f"sum.T.{name}.pengood.max"] = f"{100 * min(pg):.1f}", f"{100 * max(pg):.1f}"
        out[f"sum.T.{name}.pensilent.min"], out[f"sum.T.{name}.pensilent.max"] = f"{100 * min(ps):.1f}", f"{100 * max(ps):.1f}"
    out["sum.T.violations"] = str(viol)
    eff = [v["effectivity_median"] for v in load("bound_layered.json")["models"].values() if v["effectivity_median"]]
    out["sum.T.layered.effectivity"] = f"{min(eff):.1f}" if eff else "--"
    # X: exactly valid all-folds rule against the cross-fitted rule (cells: model, budget, variant, level)
    cells_x = undefined_more = fpr_better_eta = 0
    max_miss = {"0.05": 0.0, "0.1": 0.0}
    for name in PROBLEMS:
        af = load(f"allfolds_{name}.json")
        for k, c in af["cells"].items():
            model, r, budget, variant, rule = k.split("|")
            if variant != "allfolds":
                continue
            x = af["cells"][f"{model}|{r}|{budget}|xconf|{rule}"]
            e = af["cells"][f"{model}|{r}|{budget}|eta|{rule}"]
            cells_x += 1
            undefined_more += int(c["defined"] < x["defined"] - 0.005)
            fpr_better_eta += int(c["defined"] > 0.5 and c["fpr"] is not None and e["fpr"] is not None and c["fpr"] < e["fpr"])
            if c["miss"] is not None:
                max_miss[rule[2:]] = max(max_miss[rule[2:]], c["miss"])
    out.update({"sum.X.cells": str(cells_x), "sum.X.defless": str(undefined_more), "sum.X.beatseta": str(fpr_better_eta),
                "sum.X.maxmiss.0.05": fmt_miss(max_miss["0.05"]), "sum.X.maxmiss.0.1": fmt_miss(max_miss["0.1"])})
    # Q: graded shifts (level 0.1): worst miss rate over the shifted pools, per benchmark, indicator and guard
    for name in PROBLEMS:
        sh = load(f"shift_{name}.json")
        for variant in ("eta", "xconf"):
            for guard in ("none", "pad0.05"):
                for label, group in (("small", ("0.02", "0.05")), ("large", ("0.1", "0.15", "0.2")), ("far", ("far",))):
                    vals = [sh["cells"][f"{m}|{variant}|{guard}|cc0.1|{g}"]["miss"] for m in sh["models"] for g in group
                            if sh["cells"][f"{m}|{variant}|{guard}|cc0.1|{g}"]["miss"] is not None]
                    out[f"sum.Q.{name}.{variant}.{guard}.{label}.maxmiss"] = fmt_miss(max(vals))
                    out[f"sum.Q.{name}.{variant}.{guard}.{label}.minmiss"] = fmt_miss(min(vals))
    # Q: shift sizes and padding as written in the text (percent of the box width), from the configuration
    from contactscreen.config import SHIFTS
    pad = 5                                                  # padded guard of the text and the figure, percent
    pct = [round(100 * v) for v in SHIFTS]
    small, large = [v for v in pct if 0 < v <= pad], [v for v in pct if v > pad]
    join = lambda xs: ", ".join(map(str, xs[:-1])) + " and " + str(xs[-1])
    out.update({"sum.Q.sizes": join(pct), "sum.Q.small": join(small), "sum.Q.large": f"{large[0]} to {large[-1]}",
                "sum.Q.pad": str(pad), "sum.Q.first": str(small[0]), "sum.Q.largefrom": str(large[0]), "sum.Q.max": str(pct[-1])})
    # Q: silent-failure rate by shift size for one model per benchmark (mean over replicates)
    import numpy as np
    graded = {"membrane": "nmf.48", "block": "nmf.8", "layered": "nmf.8"}
    for name, model in graded.items():
        rates = {0.0: [], 0.02: [], 0.2: []}
        for path in sorted((RESULTS / "shift").glob(f"{name}_seed*.npz")):
            d = np.load(path)
            kind, r = model.split(".")
            silent = (d[f"{kind}|{r}|converged"] > 0.5) & (d[f"{kind}|{r}|err"] > 0.05)
            for sz in rates:
                rates[sz].append(float(silent[np.isclose(d["shift"], sz)].mean()))
        for sz, v in rates.items():
            out[f"sum.Q.{name}.silent.{sz}"] = fmt_rate(float(np.mean(v)))
    # E: sensitivity to the error tolerance (cells with the class-conditional rule defined in every split)
    n_defined, max_miss = {"0.05": 0, "0.1": 0}, {"0.05": 0.0, "0.1": 0.0}
    for name in PROBLEMS:
        ep = load(f"eps_{name}.json")
        for c in ep["cells"].values():
            for alpha in ("0.05", "0.1"):
                cell = c[f"cc{alpha}"]
                if cell["defined"] == 1.0 and cell["miss"] is not None:
                    n_defined[alpha] += 1
                    max_miss[alpha] = max(max_miss[alpha], cell["miss"])
        out[f"sum.E.{name}.ncal"] = str(ep["n_cal"])
    out["sum.E.eps.lo"], out["sum.E.eps.mid"], out["sum.E.eps.hi"] = (f"{v:g}" for v in load("eps_block.json")["eps_values"])
    for alpha in ("0.05", "0.1"):
        out[f"sum.E.defined.{alpha}"], out[f"sum.E.maxmiss.{alpha}"] = str(n_defined[alpha]), fmt_miss(max_miss[alpha])
    out["sum.E.cells"] = str(sum(len(load(f"eps_{n}.json")["cells"]) for n in PROBLEMS))
    # C: out of range, no guard, every model of a benchmark
    for name in PROBLEMS:
        o = load(f"outside_{name}.json")
        for rule in ("cc0.05", "cc0.1"):
            cells = [c for k, c in o["cells"].items() if k.endswith(f"|{rule}|none") and c["miss"] is not None]
            out[f"sum.C.{name}.{rule}.maxmiss"] = fmt_miss(max(c["miss"] for c in cells))
            out[f"sum.C.{name}.{rule}.flagged.min"] = fmt_rate(min(c["flagged"] for c in cells))
    return out


def all_numbers() -> dict:
    flat, summ = flat_numbers(), summary_numbers()
    clash = set(flat) & set(summ)
    if clash:
        raise ValueError(f"summary keys clash with flattened keys: {sorted(clash)[:5]}")
    return {**flat, **summ}
