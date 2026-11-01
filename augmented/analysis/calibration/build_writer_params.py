#!/usr/bin/env python
"""Build the TRAIN-only writer-parameter table behind timing_model="klicke_ar1".

Every value comes from the raw KLiCKe CSVs of TRAIN writers only (the split is
measure_gap.split_writers: eligible writers sorted by id, even index -> train). The output carries
no writer ids and nothing derived from held-out writers.

Per TRAIN writer, with valid IKIs defined exactly as in measure_gap (min_iki < IKI < max_iki):
  non-pause part  = valid IKIs <= pause_ms, pauses removed and order kept
      mu, sigma   mean and sd (ddof=1) of log IKI
      rho         lag-1 Pearson autocorrelation of log IKI over that sequence
  pause part      = valid IKIs > pause_ms
      p_pause     share of valid IKIs that are pauses
      pause_log_mean, pause_log_sd   mean and sd (ddof=1) of the observed log pauses
      pause_mu, pause_sigma          lognormal parameters BEFORE truncation to [pause_ms, max_iki):
                                     MLE of a doubly truncated normal on the observed log pauses.
                                     The simulator samples from that truncated law, so a drawn pause is
                                     always a pause and always survives the max_iki filter.
  n_windows       valid IKIs // window  (bootstrap weight: a window is the unit measure_gap compares)

Usage (from the package root `augmented/`):
    uv run --extra dev python analysis/calibration/build_writer_params.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import measure_gap as mg  # noqa: E402  (sibling script; shares loading, filtering and the split)

DEFAULT_OUT = PACKAGE_ROOT / "configs" / "klicke_writer_params_train.json"
COLUMNS = ["mu", "sigma", "rho", "p_pause", "pause_mu", "pause_sigma",
           "pause_log_mean", "pause_log_sd", "n_pauses", "n_windows"]
MIN_PAUSES_FOR_FIT = 20  # fewer observed pauses: use the pooled TRAIN pause law
ALPHA_MAX = 6.0          # |(a - mu) / sigma| bound of the pause fit; hitting it counts as "clamped"
SIGMA_BOUNDS = (0.2, 6.0)


def _log_mass(za: float, zb: float) -> float:
    """log(Phi(zb) - Phi(za)) for za < zb, using the tail with the smaller probabilities."""
    if za > 0:
        za, zb = -zb, -za
    lb, la = float(log_ndtr(zb)), float(log_ndtr(za))
    return lb + math.log1p(-math.exp(la - lb))


def fit_truncated_log_normal(y: np.ndarray, a: float, b: float, start=None):
    """MLE of (mu, sigma) of a normal truncated to [a, b) from samples y in [a, b).

    Parametrised by (alpha, log sigma) with mu = a - alpha * sigma so the exponential-like limit
    (mu far below a, sigma large) is reachable but bounded. Returns (mu, sigma, clamped)."""
    n = len(y)
    s1, s2 = float(y.sum()), float((y * y).sum())

    def nll(theta):
        alpha, log_sigma = theta
        sigma = math.exp(log_sigma)
        mu = a - alpha * sigma
        ss = s2 - 2.0 * mu * s1 + n * mu * mu
        return n * (log_sigma + _log_mass((a - mu) / sigma, (b - mu) / sigma)) + 0.5 * ss / (sigma * sigma)

    bounds = [(-ALPHA_MAX, ALPHA_MAX), (math.log(SIGMA_BOUNDS[0]), math.log(SIGMA_BOUNDS[1]))]
    if start is None:
        sd = float(y.std(ddof=1)) if n > 1 else 1.0
        start = (0.0, math.log(min(max(sd, SIGMA_BOUNDS[0] * 1.5), SIGMA_BOUNDS[1] / 1.5)))
    res = minimize(nll, np.asarray(start, dtype=np.float64), method="L-BFGS-B", bounds=bounds)
    alpha, log_sigma = res.x
    sigma = math.exp(log_sigma)
    clamped = bool(abs(alpha) >= ALPHA_MAX - 1e-6 or log_sigma <= bounds[1][0] + 1e-6 or log_sigma >= bounds[1][1] - 1e-6)
    return a - alpha * sigma, sigma, clamped


def writer_row(v: np.ndarray, pause_ms: float, window: int, a: float, b: float, pooled, counters) -> list:
    nonpause = v[v <= pause_ms]
    pauses = v[v > pause_ms]
    L = np.log(nonpause)
    Lp = np.log(pauses)
    if len(pauses) >= MIN_PAUSES_FOR_FIT:
        pooled_alpha = (a - pooled[0]) / pooled[1]
        mu_p, sigma_p, clamped = fit_truncated_log_normal(Lp, a, b, start=(pooled_alpha, math.log(pooled[1])))
        counters["mle"] += 1
        counters["clamped"] += int(clamped)
    else:
        mu_p, sigma_p = pooled
        counters["pooled_fallback"] += 1
    return [
        float(L.mean()), float(L.std(ddof=1)), mg._pearson(L[:-1], L[1:]),
        float(len(pauses) / len(v)), mu_p, sigma_p,
        float(Lp.mean()) if len(Lp) else float("nan"), float(Lp.std(ddof=1)) if len(Lp) > 1 else float("nan"),
        int(len(pauses)), int(len(v) // window),
    ]


def load_train_sequences(klicke_dir: Path, min_iki: float, max_iki: float, min_valid: int):
    """(train_ids, {writer_id: valid IKIs}, n_files) for the TRAIN split only.

    Pass 1 reads every file but keeps only the valid-IKI count, which defines eligibility and hence
    the split; pass 2 loads the TRAIN files. No held-out sequence is retained."""
    files = sorted(klicke_dir.glob("*.csv"), key=lambda q: q.name)
    n_valid = {f.stem: len(mg.load_writer_iki(f, min_iki, max_iki)[1]) for f in files}
    _, train_ids, _ = mg.split_writers(n_valid, min_valid)
    train = {w: mg.load_writer_iki(klicke_dir / f"{w}.csv", min_iki, max_iki)[1] for w in train_ids}
    return train_ids, train, len(files)


def parse_args(argv=None):
    d = mg.parse_args([])  # measure_gap's defaults are the single source of the filter settings
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--klicke-dir", type=Path, default=d.klicke_dir)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-iki", type=float, default=d.min_iki)
    p.add_argument("--max-iki", type=float, default=d.max_iki)
    p.add_argument("--min-valid", type=int, default=d.min_valid)
    p.add_argument("--window", type=int, default=d.window)
    p.add_argument("--pause-ms", type=float, default=d.pause_ms)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    if not any(args.klicke_dir.glob("*.csv")):
        print(f"no KLiCKe CSVs in {args.klicke_dir}", file=sys.stderr)
        return 2
    train_ids, train, n_files = load_train_sequences(args.klicke_dir, args.min_iki, args.max_iki, args.min_valid)

    a, b = math.log(args.pause_ms), math.log(args.max_iki)
    pooled_pauses = np.log(np.concatenate([v[v > args.pause_ms] for v in train.values()]))
    pooled_mu, pooled_sigma, pooled_clamped = fit_truncated_log_normal(pooled_pauses, a, b)
    pooled = (pooled_mu, pooled_sigma)

    counters = {"mle": 0, "clamped": 0, "pooled_fallback": 0}
    rows = [writer_row(train[w], args.pause_ms, args.window, a, b, pooled, counters) for w in train_ids]
    arr = np.array(rows, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        print(f"non-finite entries in columns {sorted({COLUMNS[j] for j in np.argwhere(bad)[:, 1]})}", file=sys.stderr)
        return 1

    payload = {
        "meta": {
            "description": "Writer-level timing parameters of the KLiCKe TRAIN split for timing_model='klicke_ar1'.",
            "scope": "KLiCKe is student argumentative essays typed in a browser editor, not scholarly LaTeX.",
            "built_by": "analysis/calibration/build_writer_params.py",
            "split": "TRAIN = even index of the id-sorted eligible writers (measure_gap.split_writers); "
                     "no writer ids and no held-out values are stored",
            "train_writers_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
            "n_writers": len(rows),
            "iki_filter_ms": [args.min_iki, args.max_iki],
            "min_valid_iki_per_writer": args.min_valid,
            "pause_threshold_ms": args.pause_ms,
            "window": args.window,
            "columns": {
                "mu, sigma": "mean and sd (ddof=1) of log IKI over the non-pause IKIs (<= pause_threshold_ms)",
                "rho": "lag-1 Pearson autocorrelation of log IKI over the non-pause sequence (pauses removed, order kept)",
                "p_pause": "share of valid IKIs above pause_threshold_ms",
                "pause_mu, pause_sigma": "parameters of the normal on log IKI before truncation to "
                                         "[pause_threshold_ms, iki_filter_ms[1]); doubly truncated normal MLE of the observed pauses",
                "pause_log_mean, pause_log_sd": "observed mean and sd (ddof=1) of log pauses (reference only)",
                "n_pauses": "number of observed pauses",
                "n_windows": "valid IKIs // window; the simulator draws a row with probability proportional to it",
            },
            "pooled_pause_fit": {"pause_mu": pooled_mu, "pause_sigma": pooled_sigma, "clamped": pooled_clamped,
                                 "n_pauses": int(len(pooled_pauses))},
            "pause_fit_counts": {**counters, "min_pauses_for_fit": MIN_PAUSES_FOR_FIT,
                                 "alpha_max": ALPHA_MAX, "sigma_bounds": list(SIGMA_BOUNDS)},
        },
        "columns": COLUMNS,
        "rows": [[int(x) if COLUMNS[j] in ("n_pauses", "n_windows") else round(float(x), 6)
                  for j, x in enumerate(r)] for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    print(f"TRAIN writers: {len(rows)} of {n_files} files; pooled pause fit "
          f"mu={pooled_mu:.4f} sigma={pooled_sigma:.4f} clamped={pooled_clamped}")
    print(f"per-writer pause fits: {counters}")
    for j, name in enumerate(COLUMNS):
        q = np.percentile(arr[:, j], [10, 50, 90])
        print(f"  {name:15s} p10/p50/p90 = {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
