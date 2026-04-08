"""Quantifies pause-burst dynamics as a cognitive load indicator across all datasets and task types.

Pause/burst ratio captures the planning-execution alternation fundamental to composition.
Paper relevance: provides a simple, interpretable signal for the cognitive spectrum claim.

Usage:
    uv run python analysis/extended/14_pause_burst_spectrum.py --data-dir data/ -o results/pause_burst_spectrum.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import (
    entropy_bits,
    lag1_autocorr,
    load_all,
    load_by_task_type,
    REGISTRY,
    TaskType,
)


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = _mean(x), _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else 0.0


def _kruskal_wallis(groups: List[List[float]]) -> Tuple[float, float]:
    """Kruskal-Wallis H statistic with chi-squared p-value approximation."""
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    if k < 2:
        return 0.0, 1.0

    all_vals = [v for g in groups for v in g]
    N = len(all_vals)
    if N < 3:
        return 0.0, 1.0

    # Rank all values jointly
    indexed = sorted(enumerate(all_vals), key=lambda t: t[1])
    ranks = [0.0] * N
    i = 0
    while i < N:
        j = i
        while j < N - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for kk in range(i, j + 1):
            ranks[indexed[kk][0]] = avg_rank
        i = j + 1

    # Assign ranks back to groups
    offset = 0
    H = 0.0
    for g in groups:
        ng = len(g)
        group_ranks = ranks[offset: offset + ng]
        rank_sum = sum(group_ranks)
        H += (rank_sum ** 2) / ng
        offset += ng

    H = (12.0 / (N * (N + 1))) * H - 3.0 * (N + 1)

    # Tie correction
    tie_correction = 1.0
    sorted_vals = sorted(all_vals)
    i = 0
    t_sum = 0.0
    while i < N:
        j = i
        while j < N - 1 and sorted_vals[j + 1] == sorted_vals[j]:
            j += 1
        t = j - i + 1
        if t > 1:
            t_sum += t ** 3 - t
        i = j + 1
    if t_sum > 0:
        tie_correction = 1.0 - t_sum / (N ** 3 - N)
    if tie_correction > 1e-12:
        H /= tie_correction

    # Chi-squared p-value with df = k-1 (regularized incomplete gamma approximation)
    df = k - 1
    p = _chi2_sf(H, df)
    return round(H, 6), round(p, 6)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared distribution via regularized upper incomplete gamma."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    return _upper_inc_gamma_reg(a, x / 2.0)


def _upper_inc_gamma_reg(a: float, x: float) -> float:
    """P(X > x) for Gamma(a,1) via continued fraction (Lentz method)."""
    if x < a + 1.0:
        # Use series expansion for lower incomplete gamma, then complement
        return 1.0 - _lower_inc_gamma_reg_series(a, x)
    # Continued fraction for upper incomplete gamma
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _lower_inc_gamma_reg_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma via series."""
    if x < 0:
        return 0.0
    ap = a
    s = 1.0 / a
    delta = s
    for _ in range(300):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-10:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _simulation_pause_burst(n_traces: int = 100, n_events: int = 100, seed: int = 42) -> dict:
    from scholawrite.adversarial import _generate_authentic_trace

    PAUSE_THRESHOLD_MS = 2000.0
    BURST_THRESHOLD_MS = 200.0
    ratios: List[float] = []
    for i in range(n_traces):
        trace = _generate_authentic_trace(seed + i, n_events)
        latencies = [e.latency_ms for e in trace if e.latency_ms > 0]
        if not latencies:
            continue
        pauses = sum(1 for v in latencies if v > PAUSE_THRESHOLD_MS)
        bursts = sum(1 for v in latencies if v < BURST_THRESHOLD_MS)
        ratios.append(pauses / max(bursts, 1))
    return {
        "mean_ratio": round(_mean(ratios), 6),
        "std_ratio": round(_std(ratios), 6),
        "n": len(ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/pause_burst_spectrum.json")
    parser.add_argument("--n-sim", type=int, default=100, help="Simulation traces")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _out("Loading all datasets...")
    all_datasets = load_all(data_dir)
    _out(f"  Loaded {len(all_datasets)} datasets")

    per_dataset: Dict[str, dict] = {}
    per_task_type: Dict[str, List[float]] = {}
    all_ratios: List[float] = []
    all_mean_iki: List[float] = []
    all_rev_density: List[float] = []

    for ds_name, records in all_datasets.items():
        loader = REGISTRY.get(ds_name)
        task_type = loader.METADATA.task_type.value if loader else "unknown"
        ratios = [r.pause_count / max(r.burst_count, 1) for r in records]
        if not ratios:
            continue
        per_dataset[ds_name] = {
            "mean_ratio": round(_mean(ratios), 6),
            "std_ratio": round(_std(ratios), 6),
            "n": len(ratios),
            "task_type": task_type,
        }
        per_task_type.setdefault(task_type, []).extend(ratios)

        for r, rec in zip(ratios, records):
            all_ratios.append(r)
            all_mean_iki.append(rec.mean_iki_ms)
            all_rev_density.append(rec.revision_density)

    _out(f"  Total checkpoints: {len(all_ratios)}")

    corr_iki = _pearson(all_ratios, all_mean_iki)
    corr_rev = _pearson(all_ratios, all_rev_density)
    _out(f"  Correlation with mean_iki     : {corr_iki:+.4f}")
    _out(f"  Correlation with revision_den.: {corr_rev:+.4f}")

    per_task_summary = {
        tt: {"mean_ratio": round(_mean(vals), 6), "n": len(vals)}
        for tt, vals in per_task_type.items()
    }

    # Kruskal-Wallis across task types
    task_groups = [vals for vals in per_task_type.values()]
    kw_stat, kw_p = _kruskal_wallis(task_groups)
    _out(f"  Kruskal-Wallis H={kw_stat:.4f}  p={kw_p:.4f}")

    _out("\nPer-task-type pause/burst ratio:")
    for tt in sorted(per_task_summary.keys()):
        d = per_task_summary[tt]
        _out(f"  {tt:20s}  mean={d['mean_ratio']:.4f}  n={d['n']}")

    _out(f"\nGenerating {args.n_sim} simulation traces...")
    sim_stats = _simulation_pause_burst(args.n_sim)
    _out(f"  Simulation mean ratio: {sim_stats['mean_ratio']:.4f}  std={sim_stats['std_ratio']:.4f}")

    result = {
        "per_dataset": per_dataset,
        "per_task_type": per_task_summary,
        "kruskal_wallis": {"statistic": kw_stat, "p_value": kw_p},
        "correlation_with_mean_iki": round(corr_iki, 6),
        "correlation_with_revision_density": round(corr_rev, 6),
        "simulation": sim_stats,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
