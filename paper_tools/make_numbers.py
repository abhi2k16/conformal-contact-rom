"""Write ``paper/numbers.tex``: every number the manuscript quotes, generated from the code and ``results/``.

The manuscript uses ``\\R{key}`` (defined in ``macros.tex``); an unknown key stops the build. Keys are dotted names,
for example ``block.nmf8.p`` (silent-failure rate of the NMF 8 model on ``block``). Rounding is fixed here, in
``fmt_rate`` and friends, so the text never rounds by hand.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\make_numbers.py
Inputs: ``contactscreen/config.py`` (benchmark presets), ``results/*.json``. Output: ``paper/numbers.tex``.
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contactscreen import config, problems                                        # noqa: E402
from contactscreen.conformal import EPS, min_failures                             # noqa: E402

log = logging.getLogger("make_numbers")
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "numbers.tex"
PROBLEMS = ("membrane", "block", "layered")


def fmt_rate(x: float) -> str:
    """Rates and probabilities: two decimals; three below 0.1 so that miss rates near alpha stay readable."""
    return f"{x:.3f}" if abs(x) < 0.1 else f"{x:.2f}"


def fmt_int(x) -> str:
    return f"{int(round(x)):,}".replace(",", "\\,")


def fmt_pow10(x: float) -> str:
    """Exact powers of ten as LaTeX math, for example 1e-12 -> 10^{-12} (use inside $...$)."""
    import math
    e = round(math.log10(x))
    if not math.isclose(x, 10.0 ** e):
        raise ValueError(f"{x} is not a power of ten")
    return f"10^{{{e}}}"


def fmt_ratio(x: float) -> str:
    """Speedups and factors: one decimal."""
    return f"{x:.1f}"


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the experiment that writes it")
    return json.loads(path.read_text(encoding="utf-8"))


def model_key(kind: str, r: int) -> str:
    return f"{kind}{r}"


def setting_numbers() -> dict:
    """Benchmark sizes, parameter ranges, reduced-model settings and protocol constants (Sections 2 and 5)."""
    out = {"eps": f"{EPS:g}", "nmin.0.05": fmt_int(min_failures(0.05)), "nmin.0.1": fmt_int(min_failures(0.10))}
    for name in PROBLEMS:
        prob, s = config.make_problem(name), config.SETTINGS[name]
        out.update({f"{name}.N": fmt_int(prob.n_dof), f"{name}.nc": fmt_int(prob.n_contact),
                    f"{name}.nparam": fmt_int(len(prob.lo)), f"{name}.ntrain": fmt_int(s["n_train"]),
                    f"{name}.ru": fmt_int(s["r_primal"]), f"{name}.npool": fmt_int(s["n_pool"]),
                    f"{name}.nout": fmt_int(s["n_outside"]), f"{name}.nfit": fmt_int(config.FIT_POOL[name])})
        for j, (lo, hi) in enumerate(zip(prob.lo, prob.hi)):
            out[f"{name}.lo{j}"], out[f"{name}.hi{j}"] = f"{lo:g}", f"{hi:g}"
        for attr in ("n", "stride", "d0", "kappa", "sigma", "n_xy", "nz", "depth", "nu"):
            if hasattr(prob, attr):
                out[f"{name}.{attr.replace('_', '')}"] = f"{getattr(prob, attr):g}"
        out[f"{name}.rlam"] = ", ".join(str(r) for kind, r in s["models"] if kind == "nmf")
        out[f"{name}.rsvd"] = ", ".join(str(r) for kind, r in s["models"] if kind == "svd")
    shell = {"load.lo": problems.OUT_LOAD_FACTOR[0], "load.hi": problems.OUT_LOAD_FACTOR[1], "centre.shift": problems.OUT_CENTRE_PAD,
             "ratio.lo": problems.OUT_RATIO_FACTOR[0], "ratio.hi": problems.OUT_RATIO_FACTOR[1]}
    out.update({f"out.{k}": f"{v:g}" for k, v in shell.items()})
    cov = load("coverage_membrane.json")
    out.update({"nreps.membrane": fmt_int(cov["n_replicates"]), "nsplits.coverage": fmt_int(cov["n_splits"]),
                "nboot": fmt_int(cov["n_boot"])})
    for name in ("block", "layered"):
        out[f"nreps.{name}"] = fmt_int(load(f"coverage_{name}.json")["n_replicates"])
    out["nsplits.crossfit"] = fmt_int(load("crossfit_block.json")["n_splits"])
    out["nfolds.crossfit"] = fmt_int(load("crossfit_block.json")["n_folds"])
    env = cov["environment"]
    out.update({"env.python": env["python"], "env.numpy": env["numpy"], "env.scipy": env["scipy"],
                "env.fea": env["fea_engine"], "env.rom": env["rom_engine"], "env.blas": env["blas"]["name"],
                "env.blasversion": env["blas"]["version"]})
    return out


def rate_numbers() -> dict:
    """Silent-failure rate, its interval, AUC of eta and the calibration size implied, per model (experiment B)."""
    out = {}
    for name in PROBLEMS:
        for key, r in load(f"coverage_{name}.json")["rates"].items():
            kind, dim = key.split("|")
            pre = f"{name}.{model_key(kind, int(dim))}"
            out[f"{pre}.p"] = fmt_rate(r["p"])
            out[f"{pre}.plo"], out[f"{pre}.phi"] = fmt_rate(r["p_ci"][0]), fmt_rate(r["p_ci"][1])
            if r["auc"] is not None:
                out[f"{pre}.auc"] = fmt_rate(r["auc"])
            for a in ("0.05", "0.1"):
                if r[f"queries_needed@{a}"] is not None:
                    out[f"{pre}.need{a}"] = fmt_int(r[f"queries_needed@{a}"])
    nmf = [float(v) for k, v in out.items() if ".nmf" in k and k.endswith(".p")]
    svd = [float(v) for k, v in out.items() if ".svd" in k and k.endswith(".p")]
    out.update({"rates.nmf.min": fmt_rate(min(nmf)), "rates.nmf.max": fmt_rate(max(nmf)), "rates.svd.min": fmt_rate(min(svd))})
    return out


def method_numbers() -> dict:
    """Constants of the indicators and rules (Sections 3 and 4), read from the code, and the feature cost (layered)."""
    import importlib.util
    import inspect

    from contactscreen import conformal, features, rom

    spec = importlib.util.spec_from_file_location("run_crossfit", ROOT / "scripts" / "run_crossfit.py")
    sys.path.insert(0, str(ROOT / "scripts"))
    crossfit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crossfit)
    margin = inspect.signature(conformal.heuristic_threshold).parameters["margin"].default
    fmt_list = lambda xs: ", ".join(f"{x:g}" for x in xs)
    speed = load("learned_speedup_layered.json")["models"]
    t_feat = [v["t_feat_ms_mean"] for v in speed.values()]
    t_rom = [v["t_rom_ms_mean"] for v in speed.values()]
    t_cold = [v["t_cold_ms_mean"] for v in speed.values()]
    nfeat_base = 8                                       # scalar features before the parameters, see features.feature_names
    probe = config.make_problem("membrane")
    if len(features.feature_names(probe)) != nfeat_base + 2 * len(probe.lo):
        raise ValueError("feature count changed; update method_numbers")
    return {"heur.margin": f"{margin:g}", "eta2.elambda": f"{rom.E_LAMBDA:g}", "feat.nbase": fmt_int(nfeat_base), "feat.logfloor": fmt_pow10(features.LOG_FLOOR),
            "feat.signtol": fmt_pow10(features.SIGN_TOL), "feat.ridges": fmt_list(features.RIDGES),
            "feat.nfolds": fmt_int(features.N_FOLDS), "feat.minclass": fmt_int(features.N_FOLDS),
            "xconf.nfolds": fmt_int(crossfit.N_FOLDS_XC), "xconf.ridges": fmt_list(crossfit.FAST_GRID["ridges"]),
            "xconf.selfolds": fmt_int(crossfit.FAST_GRID["n_folds"]),
            "layered.tfeat.min": f"{min(t_feat):.1f}", "layered.tfeat.max": f"{max(t_feat):.1f}",
            "layered.trom.min": f"{min(t_rom):.1f}", "layered.trom.max": f"{max(t_rom):.1f}",
            "layered.tcold": fmt_int(sum(t_cold) / len(t_cold))}


def write(numbers: dict) -> None:
    lines = ["% Generated by paper_tools/make_numbers.py from contactscreen/config.py and results/*.json. Do not edit."]
    for key in sorted(numbers):
        if "\\" in key or "{" in key:
            raise ValueError(f"bad key {key!r}")
        lines.append(f"\\expandafter\\def\\csname res@{key}\\endcsname{{{numbers[key]}}}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %d numbers to %s", len(numbers), OUT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    numbers = {}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from result_numbers import all_numbers
    for part in (setting_numbers, rate_numbers, method_numbers, all_numbers):
        new = part()
        clash = set(new) & set(numbers)
        if clash:
            raise ValueError(f"duplicate keys: {sorted(clash)}")
        numbers.update(new)
    write(numbers)


if __name__ == "__main__":
    main()
