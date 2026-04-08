"""Tests whether human writing has a thermodynamic arrow of time.

Uses increment skewness as the time-asymmetry measure. For a time-reversible
process, the distribution of increments (X_{t+1} - X_t) must be symmetric
(zero skewness). For an irreversible process, increments are skewed: fatigue
drives IKI upward over a session, creating a positive-skew increment distribution
in the forward direction. The time-reversed sequence would show negative skew.

A statistically significant non-zero skewness of IKI increments in human writers,
combined with near-zero skewness in the simulation, supports the thermodynamic
arrow claim. Paper relevance: provides physics-grounded validation of process
signals via statistical mechanics.

Note: lag-1 autocorrelation is time-reversal symmetric (ACF(x) = ACF(reverse(x)))
and therefore cannot detect the arrow of time. Increment skewness is not symmetric.

Usage:
    uv run python analysis/extended/08_thermodynamic_arrow.py \\
        --data-dir /path/to/data -o results/08_thermodynamic_arrow.json
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

from scholawrite.datasets import lag1_autocorr
from scholawrite.validation import _ks_test_2sample


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


IKI_MIN_MS = 10.0
IKI_MAX_MS = 60_000.0
MIN_EVENTS = 20


def _asymmetry(iki: List[float]) -> Optional[float]:
    """Increment skewness: skewness of (IKI[t+1] - IKI[t]) differences.

    For an irreversible process (fatigue-driven IKI increase), increments are
    positively skewed in the forward direction. Zero for time-symmetric processes.
    """
    if len(iki) < 10:
        return None
    diffs = [iki[i + 1] - iki[i] for i in range(len(iki) - 1)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / n
    if var_d < 1e-12:
        return None
    std_d = math.sqrt(var_d)
    skew = sum(((d - mean_d) / std_d) ** 3 for d in diffs) / n
    return skew


def _load_human_asymmetries(csv_dir: Path) -> List[float]:
    asymmetries: List[float] = []
    csv_files = sorted(csv_dir.glob("*.csv"))
    _out(f"  Loading {len(csv_files)} KLiCKe CSV files...")
    for idx, csv_path in enumerate(csv_files):
        if (idx + 1) % 500 == 0:
            _out(f"    {idx+1}/{len(csv_files)}...")
        iki_ms: List[float] = []
        prev_down: Optional[float] = None
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    try:
                        down_time = float(row.get("DownTime") or 0)
                    except (ValueError, TypeError):
                        continue
                    if down_time > 0:
                        if prev_down is not None and down_time > prev_down:
                            iki = down_time - prev_down
                            if IKI_MIN_MS < iki < IKI_MAX_MS:
                                iki_ms.append(iki)
                        prev_down = down_time
        except Exception:
            continue
        if len(iki_ms) < MIN_EVENTS:
            continue
        a = _asymmetry(iki_ms)
        if a is not None:
            asymmetries.append(a)
    return asymmetries


def _load_simulation_asymmetries(n_traces: int = 200, n_events: int = 100, seed: int = 42) -> List[float]:
    from scholawrite.adversarial import _generate_authentic_trace

    asymmetries: List[float] = []
    for i in range(n_traces):
        trace = _generate_authentic_trace(seed + i, n_events)
        latencies = [e.latency_ms for e in trace if e.latency_ms > 0]
        if len(latencies) < MIN_EVENTS:
            continue
        a = _asymmetry(latencies)
        if a is not None:
            asymmetries.append(a)
    return asymmetries


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, math.sqrt(var)


def _cohens_d(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    ma, sa = _mean_std(a)
    mb, sb = _mean_std(b)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2)) if na + nb > 2 else 1.0
    return (ma - mb) / pooled if pooled > 1e-12 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("08_thermodynamic_arrow.json"))
    parser.add_argument("--n-sim", type=int, default=200, help="Number of simulation traces")
    parser.add_argument("--n-events", type=int, default=100, help="Events per simulation trace")
    args = parser.parse_args()

    csv_dir = (
        args.data_dir
        / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
    )
    if not csv_dir.exists():
        _out(f"WARNING: KLiCKe CSV dir not found: {csv_dir}; skipping analysis.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"human": None, "simulation": None, "ks_statistic": None,
                   "ks_pvalue": None, "cohens_d": None}, open(args.output, "w"), indent=2)
        return

    _out("Computing human writer asymmetries...")
    human_asym = _load_human_asymmetries(csv_dir)
    _out(f"  Human writers with valid asymmetry: {len(human_asym)}")

    _out(f"Generating {args.n_sim} simulation traces ({args.n_events} events each)...")
    sim_asym = _load_simulation_asymmetries(args.n_sim, args.n_events)
    _out(f"  Simulation traces with valid asymmetry: {len(sim_asym)}")

    if not human_asym or not sim_asym:
        _out("WARNING: Empty asymmetry distribution(s); skipping comparison.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"human": {"n_writers": len(human_asym)}, "simulation": {"n_traces": len(sim_asym)},
                   "ks_statistic": None, "ks_pvalue": None, "cohens_d": None}, open(args.output, "w"), indent=2)
        return

    h_mean, h_std = _mean_std(human_asym)
    s_mean, s_std = _mean_std(sim_asym)
    h_pct_pos = sum(1 for v in human_asym if v > 0) / len(human_asym)
    s_pct_pos = sum(1 for v in sim_asym if v > 0) / len(sim_asym)

    ks_stat, ks_p = _ks_test_2sample(sorted(human_asym), sorted(sim_asym))
    d = _cohens_d(human_asym, sim_asym)

    _out(f"\nHuman  asymmetry: mean={h_mean:+.4f}  std={h_std:.4f}  pct_positive={h_pct_pos:.3f}")
    _out(f"Sim    asymmetry: mean={s_mean:+.4f}  std={s_std:.4f}  pct_positive={s_pct_pos:.3f}")
    _out(f"KS statistic: {ks_stat:.4f}  p={ks_p:.4f}")
    _out(f"Cohen's d:    {d:.4f}")

    output = {
        "human": {
            "mean_asymmetry": round(h_mean, 6),
            "std_asymmetry": round(h_std, 6),
            "pct_positive": round(h_pct_pos, 4),
            "n_writers": len(human_asym),
        },
        "simulation": {
            "mean_asymmetry": round(s_mean, 6),
            "std_asymmetry": round(s_std, 6),
            "pct_positive": round(s_pct_pos, 4),
            "n_traces": len(sim_asym),
        },
        "ks_statistic": round(ks_stat, 6),
        "ks_pvalue": round(ks_p, 6),
        "cohens_d": round(d, 6),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
    _out(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
