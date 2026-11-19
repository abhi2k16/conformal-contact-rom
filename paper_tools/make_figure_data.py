"""Compute the data of the two manuscript figures that need more than saved results (run once; ``make_figures.py`` draws).

* Silent failure (manuscript Figure "silent failure"): membrane, replicate 0, NMF 24. The query is chosen reproducibly
  as the silent failure of the audit pool whose energy error is closest to the median error of all its silent
  failures. Models are rebuilt as in the audit (``eta`` checked against the audit), the query is solved by the reduced
  and the full-order model, and the displacements and multipliers along the row of candidate contact nodes nearest the
  load centre are saved.
* ROC curves: block NMF 8 (T = 240) and layered block NMF 12 (T = 160), every replicate, one calibration/test split per
  replicate drawn with ``SEEDS['split']``. Scores on the test part: ``eta``; the learned indicator fitted on half of the
  calibration part; the cross-fitted indicator (mean of the fold models' scores, folds and search as in
  ``run_crossfit.py``). Curves are averaged over replicates vertically (true-positive rate at fixed false-positive
  rates, ``FPR_GRID``); the AUC is the mean of the replicates' AUCs. A single split is too noisy to be shown alone.

Usage, from the project folder:
    .venv\\Scripts\\python paper_tools\\make_figure_data.py
Outputs: ``results/figure_silent_failure.npz`` and ``.json``, ``results/figure_roc.npz`` and ``.json``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from contactscreen.audit import build_models, load_audit                              # noqa: E402
from contactscreen.config import RESULTS, SEEDS, audit_path, features_path, make_problem, make_settings   # noqa: E402
from contactscreen.conformal import EPS, auc, labels                                   # noqa: E402
from contactscreen.fom import FullOrderSolver, energy_error                            # noqa: E402
from contactscreen.rom import admissibility_indicator                                  # noqa: E402
from run_crossfit import N_FOLDS_XC, fit                                               # noqa: E402

log = logging.getLogger("make_figure_data")
SILENT_CASE = ("membrane", 0, ("nmf", 24))
ROC_CASES = (("block", ("nmf", 8), 240), ("layered", ("nmf", 12), 160))
FPR_GRID = np.linspace(0.0, 1.0, 201)


def silent_failure() -> None:
    name, seed, key = SILENT_CASE
    problem, settings = make_problem(name), make_settings(name, seed)
    audit = load_audit(audit_path(name, seed))
    pre = f"pool|{key[0]}|{key[1]}|"
    err, conv, eta_audit = audit.arrays[pre + "err"], audit.arrays[pre + "converged"] > 0.5, audit.arrays[pre + "eta"]
    silent = np.flatnonzero(conv & (err > EPS))
    i = int(silent[np.argmin(np.abs(err[silent] - np.median(err[silent])))])
    mu = audit.arrays["pool|mu"][i]
    solver = FullOrderSolver(problem)
    models = build_models(problem, solver, problem.sample(np.random.default_rng(settings.seed), settings.n_train), settings)
    model = models[key]
    ref = solver.solve(mu)
    red = model.solve(mu, ref.K)
    eta = admissibility_indicator(problem, mu, red)[0]
    if not np.isclose(eta, eta_audit[i], rtol=1e-8):
        raise RuntimeError("recomputed eta differs from the audit; models not rebuilt identically")
    contact = problem.B.indices
    y_rows = np.unique(np.round(problem.xy[contact, 1], 12))
    y_row = y_rows[np.argmin(np.abs(y_rows - mu[2]))]
    on_row = np.flatnonzero(np.isclose(problem.xy[:, 1], y_row))
    on_row = on_row[np.argsort(problem.xy[on_row, 0])]
    c_on_row = np.flatnonzero(np.isclose(problem.xy[contact, 1], y_row))
    c_on_row = c_on_row[np.argsort(problem.xy[contact[c_on_row], 0])]
    c = problem.offset(mu)
    gap_rom = problem.B @ red.u - c
    np.savez_compressed(RESULTS / "figure_silent_failure.npz", x=problem.xy[on_row, 0], u_fom=ref.u[on_row],
                        u_rom=red.u[on_row], xc=problem.xy[contact[c_on_row], 0], floor=c[c_on_row],
                        lam_fom=ref.lam[c_on_row], lam_rom=red.lam[c_on_row])
    meta = {"problem": name, "replicate": seed, "model": f"{key[0]}|{key[1]}", "pool_index": i, "mu": mu.tolist(),
            "y_row": float(y_row), "energy_error": float(energy_error(ref.K, ref.u, red.u)), "eta": float(eta),
            "penetration_max": float(max(0.0, -gap_rom.min())), "penetration_max_rel": float(max(0.0, -gap_rom.min()) / np.abs(c).max()),
            "penetrating_nodes": int((gap_rom < 0).sum()), "n_contact": int(len(c)),
            "active_fom": int((ref.lam > 0).sum()), "reduced_converged": bool(red.converged)}
    (RESULTS / "figure_silent_failure.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log.info("silent failure: pool query %d, error %.3f, eta %.3f", i, meta["energy_error"], meta["eta"])


def roc_points(pos: np.ndarray, neg: np.ndarray) -> tuple:
    """(false-positive rate, true-positive rate) for every threshold, flagging scores >= threshold."""
    thresholds = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = np.array([0.0] + [float((pos >= t).mean()) for t in thresholds])
    fpr = np.array([0.0] + [float((neg >= t).mean()) for t in thresholds])
    return fpr, tpr


def roc_case(name, seed, key, budget, rng) -> tuple:
    """Test-part scores of the three indicators for one replicate and one split: ({variant: (pos, neg)}, counts)."""
    data = np.load(features_path(name, seed))
    g = lambda f: data[f"pool|{key[0]}|{key[1]}|{f}"]
    lab, X = labels(g("eta"), g("err"), g("converged")), g("X")
    perm = rng.permutation(len(lab.eta))
    cal, test = perm[:budget], perm[budget:]
    scores = {"eta": lab.eta[test]}
    half = cal[: budget // 2]
    rows = half[lab.converged[half]]
    split = fit(X[rows], lab.silent[rows], int(rng.integers(1 << 30)))
    if split is None:
        raise RuntimeError(f"{name}, replicate {seed}: no split-fitted indicator could be fitted")
    scores["split"] = split.score(X[test])
    folds = rng.permutation(len(cal)) % N_FOLDS_XC
    fold_scores = []
    for k in range(N_FOLDS_XC):
        train = cal[(folds != k) & lab.converged[cal]]
        scorer = fit(X[train], lab.silent[train], int(rng.integers(1 << 30)))
        if scorer is None:
            raise RuntimeError(f"{name}, replicate {seed}: fold {k} could not be fitted")
        fold_scores.append(scorer.score(X[test]))
    scores["xconf"] = np.mean(fold_scores, axis=0)
    pos, neg = lab.silent[test], lab.good[test]
    return {v: (s_[pos], s_[neg]) for v, s_ in scores.items()}, (int(pos.sum()), int(neg.sum()))


def roc() -> None:
    rng = np.random.default_rng(SEEDS["split"])
    arrays, meta = {"fpr_grid": FPR_GRID}, {}
    for name, key, budget in ROC_CASES:
        seeds = sorted(int(p.stem.split("seed")[1]) for p in features_path(name, 0).parent.glob(f"{name}_seed*.npz"))
        curves, aucs, counts = {}, {}, []
        for seed in seeds:
            scores, n = roc_case(name, seed, key, budget, rng)
            counts.append(n)
            for variant, (pos, neg) in scores.items():
                fpr, tpr = roc_points(pos, neg)
                curves.setdefault(variant, []).append(np.interp(FPR_GRID, fpr, tpr))      # roc_points: fpr non-decreasing
                aucs.setdefault(variant, []).append(float(auc(pos, neg)))
        for variant, c in curves.items():
            arrays[f"{name}|{variant}|tpr"] = np.mean(c, axis=0)
        meta[name] = {"model": f"{key[0]}|{key[1]}", "budget": budget, "n_replicates": len(seeds),
                      "auc_mean": {v: float(np.mean(a)) for v, a in aucs.items()},
                      "auc_min": {v: float(np.min(a)) for v, a in aucs.items()},
                      "auc_max": {v: float(np.max(a)) for v, a in aucs.items()},
                      "n_silent_test_per_replicate": [c[0] for c in counts], "n_good_test_per_replicate": [c[1] for c in counts]}
        log.info("ROC %s %s: mean AUC %s", name, key, meta[name]["auc_mean"])
    np.savez_compressed(RESULTS / "figure_roc.npz", **arrays)
    (RESULTS / "figure_roc.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


BENCH_PARAMS_AT = 0.5                    # benchmark figure: parameters at the centre of the training box
THRESHOLD_CASES = (("membrane", 0, ("nmf", 24), 160, 0.10),   # threshold figure: good separation by eta
                   ("block", 0, ("nmf", 12), 160, 0.10))      # and poor separation; replicate 0, first split


def benchmarks() -> None:
    """One full-order solution per benchmark at the centre of the training box (benchmark figure)."""
    arrays, meta = {}, {}
    for name in ("membrane", "block", "layered"):
        problem = make_problem(name)
        mu = problem.lo + BENCH_PARAMS_AT * (problem.hi - problem.lo)
        sol = FullOrderSolver(problem).solve(mu)
        if not sol.converged:
            raise RuntimeError(f"{name}: full-order solve did not converge")
        contact = problem.B.indices
        xy_c = problem.xy[contact] if name == "membrane" else problem.xy_contact
        arrays[f"{name}|xy_contact"], arrays[f"{name}|active"] = xy_c, sol.active_mask.astype(float)
        arrays[f"{name}|normal_disp"] = problem.B @ sol.u
        n_full = problem.mesh.nodes.shape[0] * (1 if name == "membrane" else 3)
        u_full = np.zeros(n_full)
        u_full[problem.free] = sol.u                            # Dirichlet unknowns stay zero
        arrays[f"{name}|U_full"] = u_full
        arrays[f"{name}|contact_nodes"] = problem.free[contact] // (1 if name == "membrane" else 3)
        arrays[f"{name}|mu"] = mu
        if name == "membrane":
            arrays["membrane|xy"], arrays["membrane|u"] = problem.xy, sol.u
        meta[name] = {"mu": mu.tolist(), "param_names": list(problem.param_names), "n_active": int(sol.active_mask.sum()),
                      "n_contact": int(len(contact)), "d0": float(problem.d0), "kappa": float(problem.kappa),
                      "depth": float(getattr(problem, "depth", 0.0)), "layered": bool(getattr(problem, "layered", False))}
        log.info("benchmark %s: %d of %d candidate nodes in contact", name, meta[name]["n_active"], meta[name]["n_contact"])
    np.savez_compressed(RESULTS / "figure_benchmarks.npz", **arrays)
    (RESULTS / "figure_benchmarks.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def threshold() -> None:
    """Calibration sets of the analytical indicator and their class-conditional thresholds (threshold figure)."""
    from contactscreen.conformal import conformal_threshold

    arrays, meta = {}, {}
    for name, seed, key, n_cal, alpha in THRESHOLD_CASES:
        audit = load_audit(audit_path(name, seed))
        g = lambda f: audit.arrays[f"pool|{key[0]}|{key[1]}|{f}"]
        lab = labels(g("eta"), g("err"), g("converged"))
        cal = np.random.default_rng(SEEDS["split"]).permutation(len(lab.eta))[:n_cal]
        fail, good = lab.eta[cal][lab.silent[cal]], lab.eta[cal][lab.good[cal]]
        tau = conformal_threshold(fail, alpha)
        arrays[f"{name}|fail"], arrays[f"{name}|good"] = fail, good
        meta[name] = {"replicate": seed, "model": f"{key[0]}|{key[1]}", "n_cal": n_cal, "alpha": alpha,
                      "n_fail": int(len(fail)), "n_good": int(len(good)), "k": int(np.floor(alpha * (len(fail) + 1))),
                      "tau": float(tau), "n_fail_accepted": int((fail < tau).sum()),
                      "n_good_flagged": int((good >= tau).sum()), "share_good_flagged": float((good >= tau).mean())}
        log.info("threshold %s: %s", name, meta[name])
    np.savez_compressed(RESULTS / "figure_threshold.npz", **arrays)
    (RESULTS / "figure_threshold.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=("silent_failure", "roc", "benchmarks", "threshold"),
                        help="compute only these figures' data (default: all)")
    args = parser.parse_args()
    for part in (silent_failure, roc, benchmarks, threshold):
        if not args.only or part.__name__ in args.only:
            part()


if __name__ == "__main__":
    main()
