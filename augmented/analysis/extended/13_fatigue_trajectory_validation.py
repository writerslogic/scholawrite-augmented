"""Validates the simulation's glucose depletion model against real fatigue data.

Compares how IKI increases over session duration in real writers (Student Fatigue
dataset) vs simulation. Paper relevance: grounds the metabolic simulation in
empirical fatigue research.

Usage:
    uv run python analysis/extended/13_fatigue_trajectory_validation.py --data-dir data/ -o results/fatigue_trajectory.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr, REGISTRY, TaskType


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _linregress_slope(x: List[float], y: List[float]) -> float:
    """Returns OLS slope of y ~ x."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = _mean(x), _mean(y)
    ssxx = sum((xi - mx) ** 2 for xi in x)
    ssxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return ssxy / ssxx if ssxx > 1e-12 else 0.0


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((rxi - ryi) ** 2 for rxi, ryi in zip(rx, ry))
    denom = n * (n * n - 1)
    return 1.0 - 6.0 * d2 / denom if denom > 0 else 0.0


def _rank(vals: List[float]) -> List[float]:
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _load_fatigue_data(data_dir: Path) -> List[dict]:
    """Load student fatigue records via the registered loader."""
    from scholawrite.datasets.student_fatigue import StudentFatigueLoader
    loader = StudentFatigueLoader()
    try:
        refs = loader.load_reference(data_dir)
    except FileNotFoundError as e:
        _out(f"ERROR: {e}")
        _out("       Place dataset_kelelahan_keystroke.csv in data/student_fatigue/")
        sys.exit(1)
    return [
        {
            "participant": r.participant,
            "session": r.session,
            "iki_proxy_ms": r.mean_dwell_ms + r.mean_flight_ms,
        }
        for r in refs
        if r.mean_dwell_ms > 0 and r.mean_flight_ms > 0
    ]


def _real_fatigue_stats(records: List[dict]) -> dict:
    """Compute session-order IKI trajectory stats from fatigue data."""
    by_participant: Dict[str, List[Tuple[float, float]]] = {}
    for rec in records:
        pid = rec["participant"]
        try:
            sess_num = float(rec["session"])
        except (ValueError, TypeError):
            sess_num = 1.0
        by_participant.setdefault(pid, []).append((sess_num, rec["iki_proxy_ms"]))

    slopes: List[float] = []
    session_ikis: Dict[float, List[float]] = {}
    all_sess: List[float] = []
    all_iki: List[float] = []

    for pid, pairs in by_participant.items():
        pairs.sort(key=lambda t: t[0])
        if len(pairs) < 2:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        slope = _linregress_slope(xs, ys)
        slopes.append(slope)
        for s, v in pairs:
            session_ikis.setdefault(s, []).append(v)
            all_sess.append(s)
            all_iki.append(v)

    n_participants = len(by_participant)
    pct_increasing = sum(1 for s in slopes if s > 0) / len(slopes) if slopes else 0.0
    mean_slope = _mean(slopes)
    spearman = _spearman_rho(all_sess, all_iki)

    sess_sorted = sorted(session_ikis.keys())
    mean_iki_s1 = _mean(session_ikis.get(sess_sorted[0], [])) if sess_sorted else 0.0
    mean_iki_last = _mean(session_ikis.get(sess_sorted[-1], [])) if sess_sorted else 0.0

    return {
        "mean_iki_session1": round(mean_iki_s1, 4),
        "mean_iki_last_session": round(mean_iki_last, 4),
        "slope_ms_per_session": round(mean_slope, 6),
        "n_participants": n_participants,
        "pct_increasing": round(pct_increasing, 4),
    }, spearman


def _simulation_stats(n_traces: int = 50, n_events: int = 200, seed: int = 42) -> dict:
    """Generate simulation traces and compute IKI trajectory across 4 windows."""
    from scholawrite.adversarial import _generate_authentic_trace

    window_size = n_events // 4
    window_means: List[List[float]] = [[] for _ in range(4)]
    slopes: List[float] = []

    for i in range(n_traces):
        trace = _generate_authentic_trace(seed + i, n_events)
        latencies = [e.latency_ms for e in trace if e.latency_ms > 0]
        if len(latencies) < n_events // 2:
            continue
        windows: List[List[float]] = []
        for w in range(4):
            chunk = latencies[w * window_size: (w + 1) * window_size]
            if chunk:
                windows.append(chunk)
        if len(windows) < 2:
            continue
        w_means = [_mean(wnd) for wnd in windows]
        for idx, wm in enumerate(w_means):
            if idx < 4:
                window_means[idx].append(wm)
        xs = list(range(len(w_means)))
        slope = _linregress_slope(xs, w_means)
        slopes.append(slope)

    mean_w1 = _mean(window_means[0]) if window_means[0] else 0.0
    mean_w4 = _mean(window_means[3]) if window_means[3] else 0.0
    mean_slope = _mean(slopes)
    pct_increasing = sum(1 for s in slopes if s > 0) / len(slopes) if slopes else 0.0

    return {
        "mean_iki_window1": round(mean_w1, 4),
        "mean_iki_window4": round(mean_w4, 4),
        "slope_per_quarter": round(mean_slope, 6),
        "pct_increasing": round(pct_increasing, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/fatigue_trajectory.json")
    parser.add_argument("--n-sim", type=int, default=50, help="Simulation traces")
    parser.add_argument("--n-events", type=int, default=200, help="Events per trace")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _out("Loading Student Fatigue dataset...")
    records = _load_fatigue_data(data_dir)
    _out(f"  Records loaded: {len(records)}")

    _out("Computing real fatigue IKI trajectory...")
    real_stats, spearman = _real_fatigue_stats(records)
    _out(f"  Session 1 mean IKI  : {real_stats['mean_iki_session1']:.1f} ms")
    _out(f"  Last session mean   : {real_stats['mean_iki_last_session']:.1f} ms")
    _out(f"  Slope (ms/session)  : {real_stats['slope_ms_per_session']:+.4f}")
    _out(f"  Participants        : {real_stats['n_participants']}")
    _out(f"  Pct increasing      : {real_stats['pct_increasing']:.1%}")
    _out(f"  Spearman rho        : {spearman:.4f}")

    _out(f"\nGenerating {args.n_sim} simulation traces ({args.n_events} events each)...")
    sim_stats = _simulation_stats(args.n_sim, args.n_events)
    _out(f"  Window 1 mean IKI   : {sim_stats['mean_iki_window1']:.1f} ms")
    _out(f"  Window 4 mean IKI   : {sim_stats['mean_iki_window4']:.1f} ms")
    _out(f"  Slope (ms/quarter)  : {sim_stats['slope_per_quarter']:+.4f}")
    _out(f"  Pct increasing      : {sim_stats['pct_increasing']:.1%}")

    direction_matches = (
        real_stats["slope_ms_per_session"] > 0 and sim_stats["slope_per_quarter"] > 0
    )
    _out(f"\nDirection matches (both positive slopes): {direction_matches}")
    if direction_matches:
        _out("  RESULT: Real fatigue and simulation both show IKI increase over time,")
        _out("          consistent with glucose depletion model.")
    else:
        _out("  RESULT: Direction mismatch. Review glucose depletion parameters or")
        _out("          confirm session ordering in the fatigue dataset.")

    result = {
        "real": {**real_stats},
        "simulation": sim_stats,
        "direction_matches": direction_matches,
        "spearman_rho_real": round(spearman, 6),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
