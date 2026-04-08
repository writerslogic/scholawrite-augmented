"""Characterizes how IKI temporal structure decays with lag.

Real human writing shows slow autocorrelation decay (long-range dependence),
while AI-generated traces show fast decay (near-IID). Computes autocorrelation
at lags 1-20 for human datasets vs simulation. Paper relevance: reveals whether
human writing has long-range temporal structure that signals cognitive planning
horizons.

Usage:
    uv run python analysis/extended/15_temporal_autocorrelation_decay.py --data-dir data/ -o results/temporal_acf_decay.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr, REGISTRY, TaskType


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


IKI_MIN_MS = 10.0
IKI_MAX_MS = 60_000.0
MIN_IKI_VALUES = 100
MAX_LAG = 20
CSV_SUBDIR = Path("klicke") / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _acf_at_lag(iki: List[float], k: int) -> Optional[float]:
    """Pearson correlation between iki[:-k] and iki[k:]."""
    if k <= 0 or len(iki) <= k:
        return None
    x = iki[:-k]
    y = iki[k:]
    n = len(x)
    if n < 3:
        return None
    mx, my = _mean(x), _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def _acf_profile(iki: List[float], max_lag: int) -> List[Optional[float]]:
    return [_acf_at_lag(iki, k) for k in range(1, max_lag + 1)]


def _load_iki(csv_path: Path) -> List[float]:
    iki: List[float] = []
    prev = None
    try:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    down = float(row.get("DownTime", 0) or 0)
                except (ValueError, TypeError):
                    continue
                if prev is not None and down > prev:
                    gap = down - prev
                    if IKI_MIN_MS < gap < IKI_MAX_MS:
                        iki.append(gap)
                prev = down
    except Exception:
        pass
    return iki


def _mean_acf(profiles: List[List[Optional[float]]], max_lag: int) -> List[float]:
    """Mean ACF across writers at each lag, ignoring None values."""
    result = []
    for k_idx in range(max_lag):
        vals = [p[k_idx] for p in profiles if p[k_idx] is not None]
        result.append(_mean(vals) if vals else 0.0)
    return result


def _half_life(acf: List[float]) -> float:
    """Lag at which ACF drops to half its lag-1 value. Interpolates between lags."""
    if not acf or acf[0] <= 0:
        return 1.0
    target = acf[0] / 2.0
    for i in range(1, len(acf)):
        if acf[i] <= target:
            if acf[i - 1] > acf[i]:
                frac = (acf[i - 1] - target) / (acf[i - 1] - acf[i])
                return (i - 1) + frac + 1.0
            return float(i + 1)
    return float(len(acf) + 1)


def _t_test_vs_zero(vals: List[float]) -> Tuple[float, float]:
    """One-sample t-test of vals against zero. Returns (t, p_two_tailed)."""
    n = len(vals)
    if n < 2:
        return 0.0, 1.0
    m = _mean(vals)
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    if se < 1e-12:
        return 0.0, 1.0
    t = m / se
    # Two-tailed p-value approximation via t-distribution CDF (Abramowitz & Stegun)
    df = n - 1
    p = _t_sf(abs(t), df) * 2.0
    return round(t, 6), round(min(p, 1.0), 6)


def _t_sf(t: float, df: int) -> float:
    """Survival function P(T > t) for t-distribution via regularized incomplete beta."""
    if t < 0:
        return 1.0
    x = df / (df + t * t)
    return 0.5 * _betainc_reg(df / 2.0, 0.5, x)


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    # Use CF for x < (a+1)/(a+b+2), else use symmetry
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/temporal_acf_decay.json")
    parser.add_argument("--n-sim", type=int, default=100, help="Simulation traces")
    parser.add_argument("--n-events", type=int, default=200, help="Events per simulation trace")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_dir = data_dir / CSV_SUBDIR
    if not csv_dir.exists():
        _out(f"ERROR: KLiCKe CSV directory not found at {csv_dir}")
        _out("       Download KLiCKe with: uv run python -c \"from scholawrite.datasets.klicke import KLiCKeLoader; "
             "KLiCKeLoader().download(Path('data'))\"")
        sys.exit(1)

    csv_files = sorted(csv_dir.glob("*.csv"))
    _out(f"Found {len(csv_files)} KLiCKe writer CSVs")

    human_profiles: List[List[Optional[float]]] = []
    n_skipped = 0
    for csv_path in csv_files:
        iki = _load_iki(csv_path)
        if len(iki) < MIN_IKI_VALUES:
            n_skipped += 1
            continue
        profile = _acf_profile(iki, MAX_LAG)
        if profile[0] is not None:
            human_profiles.append(profile)
        if len(human_profiles) % 500 == 0 and len(human_profiles) > 0:
            _out(f"  Processed {len(human_profiles)} writers...")

    n_writers = len(human_profiles)
    _out(f"  Writers with >= {MIN_IKI_VALUES} IKI values: {n_writers}  (skipped {n_skipped})")

    if n_writers == 0:
        _out("ERROR: No writers with sufficient IKI data.")
        sys.exit(1)

    human_mean_acf = _mean_acf(human_profiles, MAX_LAG)

    _out(f"\nGenerating {args.n_sim} simulation traces ({args.n_events} events each)...")
    from scholawrite.adversarial import _generate_authentic_trace

    sim_profiles: List[List[Optional[float]]] = []
    for i in range(args.n_sim):
        trace = _generate_authentic_trace(42 + i, args.n_events)
        latencies = [e.latency_ms for e in trace if e.latency_ms > 0]
        if len(latencies) < MIN_IKI_VALUES:
            continue
        profile = _acf_profile(latencies, MAX_LAG)
        if profile[0] is not None:
            sim_profiles.append(profile)

    _out(f"  Simulation traces with valid ACF: {len(sim_profiles)}")

    sim_mean_acf = _mean_acf(sim_profiles, MAX_LAG)

    human_half_life = _half_life(human_mean_acf)
    sim_half_life = _half_life(sim_mean_acf)
    _out(f"\nHuman   ACF half-life lag : {human_half_life:.2f}")
    _out(f"Sim     ACF half-life lag : {sim_half_life:.2f}")

    # LRD test at lag 20: collect per-writer ACF at lag 20, t-test vs 0
    human_lag20_vals = [
        p[MAX_LAG - 1] for p in human_profiles if p[MAX_LAG - 1] is not None
    ]
    sim_lag20_vals = [
        p[MAX_LAG - 1] for p in sim_profiles if p[MAX_LAG - 1] is not None
    ]

    h_t, h_p = _t_test_vs_zero(human_lag20_vals)
    s_t, s_p = _t_test_vs_zero(sim_lag20_vals)
    h_lrd_sig = h_p < 0.05 and _mean(human_lag20_vals) > 0
    s_lrd_sig = s_p < 0.05 and _mean(sim_lag20_vals) > 0

    _out(f"\nHuman LRD at lag {MAX_LAG}: mean={_mean(human_lag20_vals):+.4f}  t={h_t:.3f}  p={h_p:.4f}  significant={h_lrd_sig}")
    _out(f"Sim   LRD at lag {MAX_LAG}: mean={_mean(sim_lag20_vals):+.4f}  t={s_t:.3f}  p={s_p:.4f}  significant={s_lrd_sig}")

    _out("\nHuman mean ACF by lag:")
    for k, v in enumerate(human_mean_acf, start=1):
        _out(f"  lag {k:2d}: {v:+.4f}")
    _out("\nSimulation mean ACF by lag:")
    for k, v in enumerate(sim_mean_acf, start=1):
        _out(f"  lag {k:2d}: {v:+.4f}")

    result = {
        "human_acf": [round(v, 6) for v in human_mean_acf],
        "simulation_acf": [round(v, 6) for v in sim_mean_acf],
        "human_half_life_lag": round(human_half_life, 4),
        "simulation_half_life_lag": round(sim_half_life, 4),
        "human_lrd_at_lag20": {
            "mean_acf": round(_mean(human_lag20_vals), 6),
            "t_statistic": h_t,
            "p_value": h_p,
            "significant": h_lrd_sig,
        },
        "simulation_lrd_at_lag20": {
            "mean_acf": round(_mean(sim_lag20_vals), 6),
            "p_value": s_p,
            "significant": s_lrd_sig,
        },
        "n_writers": n_writers,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
