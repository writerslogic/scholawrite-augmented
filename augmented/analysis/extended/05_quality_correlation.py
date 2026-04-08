"""Correlates process signals with writing quality scores (KLiCKe holistic scores).

Tests whether keystroke dynamics predict essay quality -- something no text-only
detector can measure. Paper relevance: demonstrates process signal utility beyond
AI detection. Spearman correlations with bootstrap 95% CI are computed for each
signal. The top 3 predictors are reported.

Usage:
    uv run python analysis/extended/05_quality_correlation.py \\
        --data-dir /path/to/data -o results/05_quality_correlation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


# ---------------------------------------------------------------------------
# IKI feature extraction
# ---------------------------------------------------------------------------

IKI_MIN_MS = 10.0
IKI_MAX_MS = 60_000.0
MIN_EVENTS = 20


def _compute_writer_features(csv_path: Path) -> Optional[Dict[str, float]]:
    """Parse one KLiCKe CSV and return a dict of per-writer features, or None."""
    iki_ms: List[float] = []
    input_chars = 0
    remove_events = 0
    nonprod_events = 0
    total_events = 0
    prev_down: Optional[float] = None
    first_down: Optional[float] = None
    last_down: Optional[float] = None

    try:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    down_time = float(row.get("DownTime") or 0)
                except (ValueError, TypeError):
                    continue

                total_events += 1
                activity = (row.get("Activity") or "").strip()
                text_change = (row.get("TextChange") or "").strip()

                if activity == "Remove/Cut":
                    remove_events += 1
                elif activity == "Nonproduction":
                    nonprod_events += 1
                elif activity == "Input" and text_change and text_change != "NoChange":
                    input_chars += len(text_change)

                if down_time > 0:
                    if first_down is None:
                        first_down = down_time
                    last_down = down_time
                    if prev_down is not None and down_time > prev_down:
                        iki = down_time - prev_down
                        if IKI_MIN_MS < iki < IKI_MAX_MS:
                            iki_ms.append(iki)
                    prev_down = down_time
    except Exception:
        return None

    if total_events < MIN_EVENTS or len(iki_ms) < MIN_EVENTS:
        return None

    mean_iki = sum(iki_ms) / len(iki_ms)
    variance = sum((x - mean_iki) ** 2 for x in iki_ms) / len(iki_ms)
    std_iki = math.sqrt(variance)
    ent = entropy_bits(iki_ms)
    ac = lag1_autocorr(iki_ms) or 0.0
    revision_density = remove_events / total_events if total_events > 0 else 0.0
    planning_ratio = nonprod_events / total_events if total_events > 0 else 0.0
    burst_count = sum(1 for x in iki_ms if x < 200.0)
    pause_count = sum(1 for x in iki_ms if x > 2000.0)

    total_time_ms = (last_down - first_down) if (last_down and first_down) else sum(iki_ms)
    wpm = (input_chars / 5.0) / (total_time_ms / 60_000.0) if total_time_ms > 0 else 0.0

    return {
        "mean_iki": mean_iki,
        "std_iki": std_iki,
        "entropy_bits": ent,
        "lag1_autocorr": ac,
        "revision_density": revision_density,
        "planning_ratio": planning_ratio,
        "wpm": wpm,
        "burst_count": float(burst_count),
        "pause_count": float(pause_count),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _rank(values: List[float]) -> List[float]:
    """Return ranks (1-based, average ties)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_pvalue(rho: float, n: int) -> float:
    """Two-tailed p-value via t-distribution approximation."""
    if n < 3:
        return 1.0
    if abs(rho) >= 1.0:
        return 0.0
    t_stat = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    # Two-tailed p via normal approximation (valid for large n)
    # Use beta-incomplete function approximation for t-dist
    df = n - 2
    x = df / (df + t_stat * t_stat)
    # Regularized incomplete beta approximation
    p_one_tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return min(1.0, 2.0 * p_one_tail)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta via continued fraction (Lentz method)."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    # Lentz continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, 200):
        # Even step
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        c = 1.0 + num / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c
        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        c = 1.0 + num / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * f


def _bootstrap_rho_ci(
    x: List[float],
    y: List[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for Spearman rho."""
    rng = random.Random(seed)
    n = len(x)
    boot_rhos: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        boot_rhos.append(_spearman_rho(bx, by))
    boot_rhos.sort()
    lo = boot_rhos[int(0.025 * n_boot)]
    hi = boot_rhos[int(0.975 * n_boot)]
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("05_quality_correlation.json"))
    args = parser.parse_args()

    csv_dir = args.data_dir / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
    scores_file = args.data_dir / "klicke" / "Files" / "WritingTask" / "WritingTask" / "holistic_scores.csv"

    if not csv_dir.exists():
        _out(f"WARNING: KLiCKe CSV dir not found: {csv_dir}; skipping analysis.")
        json.dump({"n_writers": 0, "signals": {}, "top_predictors": []}, open(args.output, "w"), indent=2)
        return
    if not scores_file.exists():
        _out(f"WARNING: Holistic scores file not found: {scores_file}; skipping analysis.")
        json.dump({"n_writers": 0, "signals": {}, "top_predictors": []}, open(args.output, "w"), indent=2)
        return

    # Load quality scores
    _out("Loading holistic scores...")
    scores: Dict[str, float] = {}
    with open(scores_file, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            writer_id = str(row.get("ID", "")).strip()
            try:
                score = float(row.get("Score", 0))
            except (ValueError, TypeError):
                continue
            if writer_id:
                scores[writer_id] = score
    _out(f"  Loaded {len(scores)} quality scores")

    # Load keystroke CSVs
    csv_files = sorted(csv_dir.glob("*.csv"))
    _out(f"Processing {len(csv_files)} writer CSV files...")
    feature_rows: List[Dict] = []
    skipped = 0
    for i, csv_path in enumerate(csv_files):
        if (i + 1) % 500 == 0:
            _out(f"  {i+1}/{len(csv_files)}...")
        writer_id = csv_path.stem
        if writer_id not in scores:
            skipped += 1
            continue
        feats = _compute_writer_features(csv_path)
        if feats is None:
            skipped += 1
            continue
        feats["writer_id"] = writer_id
        feats["score"] = scores[writer_id]
        feature_rows.append(feats)

    n_writers = len(feature_rows)
    _out(f"  Usable writers: {n_writers} (skipped {skipped})")

    if n_writers < 10:
        _out(f"WARNING: Too few writers with matching scores ({n_writers}); skipping analysis.")
        json.dump({"n_writers": n_writers, "signals": {}, "top_predictors": []}, open(args.output, "w"), indent=2)
        return

    signal_names = ["mean_iki", "std_iki", "entropy_bits", "lag1_autocorr",
                    "revision_density", "planning_ratio", "wpm", "burst_count", "pause_count"]
    quality_scores = [r["score"] for r in feature_rows]

    results_signals: Dict[str, Dict] = {}
    _out("\nComputing Spearman correlations with bootstrap 95% CI...")
    for sig in signal_names:
        sig_vals = [r[sig] for r in feature_rows]
        rho = _spearman_rho(sig_vals, quality_scores)
        p = _spearman_pvalue(rho, n_writers)
        ci_lo, ci_hi = _bootstrap_rho_ci(sig_vals, quality_scores)
        results_signals[sig] = {
            "spearman_rho": round(rho, 4),
            "p_value": round(p, 6),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        }
        _out(f"  {sig:20s}: rho={rho:+.3f}  p={p:.4f}  95%CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]")

    # Top 3 by |rho|
    sorted_sigs = sorted(signal_names, key=lambda s: abs(results_signals[s]["spearman_rho"]), reverse=True)
    top_3 = sorted_sigs[:3]
    _out(f"\nTop 3 predictors of quality: {top_3}")

    output = {
        "n_writers": n_writers,
        "signals": results_signals,
        "top_predictors": top_3,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
    _out(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
