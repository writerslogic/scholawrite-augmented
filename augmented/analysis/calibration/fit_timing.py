#!/usr/bin/env python
"""TRAIN-only fit of the timing_model="klicke_ar1" parameters.

Fitted: timing_offset, timing_rho_scale, timing_sigma_scale, timing_ar_share.

Target = test estimator. The decision rule compares 50-IKI window statistics, so the target is the
set of window statistics of TRAIN writers (measure_gap's estimators and filters, consecutive
non-overlapping windows), not the whole-sequence writer table. The simulated side runs end to end
through scholawrite.adversarial._generate_authentic_trace, the path the experiments use.

Stage 1 (quantile match): Nelder-Mead on
    sum over the four decision metrics of
    ((median_sim - median_train) / IQR_train)^2 + ((IQR_sim - IQR_train) / IQR_train)^2
plus, for pause_fraction (a discrete variable with coarse quartiles), the same scaled squared
difference of its mean.
Stage 2 (KS polish): Nelder-Mead from the stage-1 optimum on the sum over the four decision
metrics of the squared two-sample KS D against ALL TRAIN windows. Stage 1 alone leaves the
decision statistic fragile (its D varies by ~0.02 across independent simulation samples because
medians and IQRs do not pin the CDF), so the rule's own statistic is minimised on TRAIN as well.

Fit seeds (500000+i, 600000+i) differ from the reference seeds 10000+i used by measure_gap;
replicate sets at other seeds measure how much D moves with simulation noise. Held-out writers are
never analysed: only TRAIN sequences are retained (build_writer_params.load_train_sequences).

Usage (from the package root `augmented/`):
    uv run --extra dev python analysis/calibration/fit_timing.py
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import ks_2samp

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import build_writer_params as bwp  # noqa: E402
import measure_gap as mg  # noqa: E402

DEFAULT_OUT = SCRIPT_DIR / "fit_timing_result.json"
DECISION = mg.DECISION_METRICS
START = {"timing_offset": -0.8, "timing_rho_scale": 1.0, "timing_sigma_scale": 1.0, "timing_ar_share": 0.5}
STEPS = np.array([0.3, 0.4, 0.3, 0.6])  # initial simplex edge in (offset, ln rho_scale, ln sigma_scale, logit share)
REPLICATE_SEED_BASES = (700000, 800000, 900000, 10000)  # the last one is the measure_gap reference set
ROUND = 4  # the parameters written to the config are rounded; every reported figure uses the rounded values

_W: dict = {}


# ---------------------------------------------------------------------------
# Simulator workers (one process per core; each imports the working-tree package)
# ---------------------------------------------------------------------------

def _init_worker(root: str) -> None:
    causal_core, embodied = mg.import_simulator(Path(root))
    _, gen = mg.resolve_trace_generator(causal_core, embodied)
    _W.update(cc=causal_core, emb=embodied, gen=gen, base=embodied.get_sim_config())


def _simulate_chunk(job):
    overrides, seeds, n_events = job
    mg.apply_sim_overrides(_W["cc"], _W["emb"], overrides, base=_W["base"])
    return [np.asarray(_W["gen"](s, n_events)[0], dtype=np.float64) for s in seeds]


def simulate(pool, overrides: dict, seed_base: int, n_traces: int, n_events: int, n_chunks: int):
    bounds = np.linspace(0, n_traces, n_chunks + 1).astype(int)
    jobs = [(overrides, list(range(seed_base + a, seed_base + b)), n_events)
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return [seq for part in pool.map(_simulate_chunk, jobs) for seq in part]


# ---------------------------------------------------------------------------
# Window summaries, loss, KS
# ---------------------------------------------------------------------------

def sim_windows(seqs, args) -> dict:
    filt = [mg.filter_valid(x, args.min_iki, args.max_iki) for x in seqs]
    return mg.window_metrics(filt, args.window, args.pause_ms, args.burst_ms, args.burst_min_run, args.bin_ms)


def summarize(wm: dict) -> dict:
    """{decision metric: median/q25/q75/iqr/mean/n (+ p0, the share of pause-free windows, for pause_fraction)}."""
    out = {}
    for k in DECISION:
        v = wm[k][~np.isnan(wm[k])]
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        out[k] = {"median": float(q50), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
                  "mean": float(v.mean()), "n": int(len(v))}
    out["pause_fraction"]["p0"] = float((wm["pause_fraction"] == 0).mean())
    return out


def quantile_loss(sim: dict, tgt: dict) -> float:
    total = 0.0
    for k in DECISION:
        s, t = sim[k], tgt[k]
        total += ((s["median"] - t["median"]) / t["iqr"]) ** 2 + ((s["iqr"] - t["iqr"]) / t["iqr"]) ** 2
    s, t = sim["pause_fraction"], tgt["pause_fraction"]
    return total + ((s["mean"] - t["mean"]) / t["iqr"]) ** 2


def ks_values(wm: dict, train_vals: dict) -> dict:
    return {k: float(ks_2samp(train_vals[k], wm[k][~np.isnan(wm[k])]).statistic) for k in DECISION}


def to_overrides(x, digits=None) -> dict:
    """Unconstrained vector -> SimulationConfig overrides for the klicke_ar1 model."""
    p = {
        "timing_model": "klicke_ar1",
        "timing_offset": float(x[0]),
        "timing_rho_scale": float(math.exp(x[1])),
        "timing_sigma_scale": float(math.exp(x[2])),
        "timing_ar_share": float(1.0 / (1.0 + math.exp(-x[3]))),
    }
    if digits is not None:
        p = {k: (round(v, digits) if isinstance(v, float) else v) for k, v in p.items()}
    return p


def from_params(p: dict):
    s = min(max(p["timing_ar_share"], 1e-6), 1 - 1e-6)
    return np.array([p["timing_offset"], math.log(p["timing_rho_scale"]), math.log(p["timing_sigma_scale"]),
                     math.log(s / (1.0 - s))])


def nelder_mead(fun, x0, steps, max_evals, passes, history):
    """Restarted Nelder-Mead; stops when a pass no longer improves. `fun` records itself in `history`."""
    best_x, best_f = np.asarray(x0, dtype=np.float64), fun(x0)
    for p in range(passes):
        left = max_evals - len(history)
        if left <= 8:
            break
        simplex = np.vstack([best_x] + [best_x + np.eye(len(best_x))[i] * steps[i] * (0.5 ** p)
                                        for i in range(len(best_x))])
        res = minimize(fun, best_x, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "maxfev": left, "xatol": 1e-3, "fatol": 1e-6})
        print(f"    pass {p + 1}: f={res.fun:.6f} evals={len(history)}")
        if res.fun < best_f - 1e-9:
            best_x, best_f = np.asarray(res.x, dtype=np.float64), float(res.fun)
        else:
            break
    return best_x, best_f


# ---------------------------------------------------------------------------
# TRAIN gap through the measure_gap CLI (the same procedure as the held-out comparison)
# ---------------------------------------------------------------------------

def gap_train(overrides: dict, tmp: Path, tag: str) -> dict:
    out = tmp / f"gap_train_{tag}.json"
    cmd = [sys.executable, str(SCRIPT_DIR / "measure_gap.py"), "--split", "train", "--out", str(out),
           "--writer-params-out", str(tmp / f"wp_{tag}.json")]
    for k, v in overrides.items():
        cmd += ["--sim-override", f"{k}={json.dumps(v)}"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


def gap_table(gap: dict) -> dict:
    return {m: {"klicke": {q: v["klicke"][q] for q in ("q25", "median", "q75")},
                "simulator": {q: v["simulator"][q] for q in ("q25", "median", "q75")},
                "ks_D": v["ks_D"]} for m, v in gap["window_metrics"].items()}


def window_position_diagnostic(wm_by_trace: dict) -> dict:
    """Decision metrics of window 0 (glucose still falling) vs the later windows."""
    return {k: {"window0": mg.quartiles(v[:, 0]), "windows1plus": mg.quartiles(v[:, 1:].ravel())}
            for k, v in wm_by_trace.items()}


def parse_args(argv=None):
    d = mg.parse_args([])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--klicke-dir", type=Path, default=d.klicke_dir)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stage1-seed-base", type=int, default=500000)
    p.add_argument("--stage2-seed-base", type=int, default=600000)
    p.add_argument("--stage1-traces", type=int, default=1000)
    p.add_argument("--stage2-traces", type=int, default=3000)
    p.add_argument("--n-events", type=int, default=200)
    p.add_argument("--stage1-evals", type=int, default=200)
    p.add_argument("--stage2-evals", type=int, default=110)
    p.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    p.add_argument("--skip-train-gap", action="store_true", help="skip the final measure_gap --split train runs")
    for name in ("min_iki", "max_iki", "min_valid", "window", "pause_ms", "burst_ms", "burst_min_run", "bin_ms"):
        p.add_argument("--" + name.replace("_", "-"), type=type(getattr(d, name)), default=getattr(d, name))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    train_ids, train, _ = bwp.load_train_sequences(args.klicke_dir, args.min_iki, args.max_iki, args.min_valid)
    train_wm = mg.window_metrics([train[w] for w in train_ids], args.window, args.pause_ms,
                                 args.burst_ms, args.burst_min_run, args.bin_ms)
    target = summarize(train_wm)
    train_vals = {k: train_wm[k][~np.isnan(train_wm[k])] for k in DECISION}
    del train
    n_train_windows = target["mean_iki_ms"]["n"]
    print(f"TRAIN: {len(train_ids)} writers, {n_train_windows} windows ({time.time() - t0:.0f}s)")

    n_chunks = args.workers * 3
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(str(PACKAGE_ROOT),)) as pool:

        def run(x, seed_base, n_traces):
            return sim_windows(simulate(pool, to_overrides(x), seed_base, n_traces, args.n_events, n_chunks), args)

        # ---- stage 1: quantile match ------------------------------------
        hist1: list = []

        def f1(x):
            v = quantile_loss(summarize(run(x, args.stage1_seed_base, args.stage1_traces)), target)
            hist1.append(v)
            return v

        print("stage 1 (median/IQR match)")
        x1, loss1 = nelder_mead(f1, from_params(START), STEPS, args.stage1_evals, 3, hist1)
        print(f"  stage 1: {to_overrides(x1, ROUND)} loss={loss1:.5f} ({len(hist1)} evaluations)")

        # ---- stage 2: KS polish ----------------------------------------
        hist2: list = []

        def f2(x):
            v = sum(d * d for d in ks_values(run(x, args.stage2_seed_base, args.stage2_traces), train_vals).values())
            hist2.append(v)
            return v

        print("stage 2 (squared TRAIN KS D)")
        x2, loss2 = nelder_mead(f2, x1, STEPS * 0.5, args.stage2_evals, 2, hist2)
        # evaluate the parameters exactly as they will be written to the config
        fitted = to_overrides(x2, ROUND)
        x_final = from_params(fitted)
        print(f"  stage 2: {fitted} sum D^2={loss2:.5f} ({len(hist2)} evaluations)")

        # ---- diagnostics at the written parameters -------------------------
        stage1_params = to_overrides(x1, ROUND)
        fit_wm = run(x_final, args.stage1_seed_base, args.stage1_traces)
        s1_wm = run(from_params(stage1_params), args.stage1_seed_base, args.stage1_traces)
        replicates = {}
        for base in REPLICATE_SEED_BASES:
            for tag, params in (("stage1", stage1_params), ("fitted", fitted)):
                wm = sim_windows(simulate(pool, params, base, 1000, args.n_events, n_chunks), args)
                replicates.setdefault(str(base), {})[tag] = ks_values(wm, train_vals)
        ref_wm_seqs = sim_windows(simulate(pool, fitted, 10000, 1000, args.n_events, n_chunks), args)
        position = window_position_diagnostic(
            {k: ref_wm_seqs[k].reshape(1000, -1) for k in DECISION}) if len(ref_wm_seqs["mean_iki_ms"]) % 1000 == 0 else {}

    result = {
        "meta": {
            "script": "analysis/calibration/fit_timing.py",
            "scope": "calibrated to KLiCKe keystroke timing (student argumentative essays), fitted on TRAIN writers only",
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "stage1": {"seed_base": args.stage1_seed_base, "n_traces": args.stage1_traces,
                       "n_evaluations": len(hist1), "objective": "scaled squared differences of window median and IQR "
                       "(scale = TRAIN IQR) for the four decision metrics, plus the mean of pause_fraction"},
            "stage2": {"seed_base": args.stage2_seed_base, "n_traces": args.stage2_traces,
                       "n_evaluations": len(hist2), "objective": "sum of squared KS D against all TRAIN windows, four decision metrics"},
            "optimizer": "restarted Nelder-Mead on (offset, ln rho_scale, ln sigma_scale, logit ar_share), common random numbers",
            "n_events": args.n_events, "window": args.window, "n_train_writers": len(train_ids),
            "n_train_windows": n_train_windows, "workers": args.workers, "rounded_to_decimals": ROUND,
        },
        "start": START, "stage1_params": stage1_params, "fitted": fitted,
        "stage1_loss": loss1, "stage2_sum_D2": loss2,
        "train_target": target,
        "quantile_residuals_in_train_iqr_units": {
            tag: {k: {"median": (summarize(w)[k]["median"] - target[k]["median"]) / target[k]["iqr"],
                      "iqr": (summarize(w)[k]["iqr"] - target[k]["iqr"]) / target[k]["iqr"]} for k in DECISION}
            for tag, w in (("stage1", s1_wm), ("fitted", fit_wm))},
        "pause_free_window_share": {"train": target["pause_fraction"]["p0"],
                                    "fitted_sim": summarize(fit_wm)["pause_fraction"]["p0"]},
        "ks_D_vs_all_train_windows_by_simulation_sample": replicates,
        "window_position_diagnostic_reference_seeds": position,
    }
    four_param_reaches_train_gate = all(max(v["fitted"].values()) <= mg.DECISION_KS_MAX for v in replicates.values())
    result["drift_term"] = {
        "added": False,
        "four_parameter_fit_passes_train_gate_on_every_simulation_sample": bool(four_param_reaches_train_gate),
        "rule": "timing_drift_sd / timing_drift_rho are only to be added if the four-parameter version cannot reach "
                "the TRAIN targets; here the TRAIN KS gate holds on all replicate samples, so they were not added",
    }
    if not args.skip_train_gap:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            legacy = gap_train({}, tmp, "legacy")
            fitted_gap = gap_train(fitted, tmp, "fitted")
        result["train_gap"] = {"legacy": gap_table(legacy), "fitted": gap_table(fitted_gap)}
        d_fit = {m: result["train_gap"]["fitted"][m]["ks_D"] for m in mg.METRICS}
        result["train_gate"] = {"metrics": DECISION, "ks_max": mg.DECISION_KS_MAX,
                                "ks_D": {m: d_fit[m] for m in DECISION},
                                "pass": bool(max(d_fit[m] for m in DECISION) <= mg.DECISION_KS_MAX)}
        print("\n| metric | TRAIN median [IQR] | legacy sim median [IQR], KS D | fitted sim median [IQR], KS D |")
        print("|---|---|---|---|")
        for m in mg.METRICS:
            a, b = result["train_gap"]["legacy"][m], result["train_gap"]["fitted"][m]
            print(f"| {m} | {_fmt(b['klicke'])} | {_fmt(a['simulator'])}, {a['ks_D']:.3f} | "
                  f"{_fmt(b['simulator'])}, {b['ks_D']:.3f} |")
        print(f"\nTRAIN gate on the reference seeds (KS D <= {mg.DECISION_KS_MAX} on {DECISION}): "
              f"{'PASS' if result['train_gate']['pass'] else 'FAIL'}")
    print("\nKS D vs all TRAIN windows, by simulation sample (stage 1 / fitted):")
    for base, v in replicates.items():
        print(f"  seeds {base}+i  " + "  ".join(
            f"{t}: " + " ".join(f"{v[t][k]:.3f}" for k in DECISION) for t in ("stage1", "fitted")))
    print(f"\nfitted parameters: {fitted}")
    result["meta"]["runtime_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mg._clean(result), indent=2) + "\n")
    print(f"Wrote {args.out}  (runtime {time.time() - t0:.0f}s)")
    return 0


def _fmt(q: dict) -> str:
    return f"{q['median']:.4g} [{q['q25']:.4g}, {q['q75']:.4g}]"


if __name__ == "__main__":
    sys.exit(main())
