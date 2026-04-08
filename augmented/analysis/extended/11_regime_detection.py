"""Detects cognitive regime transitions within documents.

Uses rolling IKI entropy to identify when a writer switches between composition
(high entropy, planning-intensive) and transcription/recall (low entropy, automatic).

Paper relevance: demonstrates within-document cognitive state discrimination at
keystroke granularity.

Usage:
    uv run python analysis/extended/11_regime_detection.py --data-dir data/ -o results/regime_detection.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr, load_all, REGISTRY, TaskType
from scholawrite.validation import generate_simulation_reference

CSV_SUBDIR = Path("klicke") / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"

IKI_MIN_MS = 10.0
IKI_MAX_MS = 60_000.0
MIN_IKI_VALUES = 100
WINDOW = 50
STRIDE = 10
MIN_TRANSITIONS = 2
MIN_REGIME_PCT = 0.20


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _load_iki(csv_path: Path) -> list[float]:
    iki: list[float] = []
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
    except OSError:
        pass
    return iki


def _rolling_entropy(iki: list[float], window: int, stride: int) -> list[float]:
    result = []
    i = 0
    while i + window <= len(iki):
        result.append(entropy_bits(iki[i : i + window]))
        i += stride
    return result


def _percentile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _analyze_writer(
    iki: list[float], high_thresh: float, low_thresh: float
) -> dict | None:
    """Return regime-switching profile for one writer, or None if insufficient data."""
    rolling = _rolling_entropy(iki, WINDOW, STRIDE)
    if len(rolling) < 4:
        return None

    labels: list[str] = []
    for v in rolling:
        if v >= high_thresh:
            labels.append("HIGH")
        elif v <= low_thresh:
            labels.append("LOW")
        else:
            labels.append("MID")

    n_high = labels.count("HIGH")
    n_low = labels.count("LOW")
    n_total = len(labels)

    pct_high = n_high / n_total
    pct_low = n_low / n_total

    # Count HIGH<->LOW transitions (ignore MID windows).
    filtered = [l for l in labels if l != "MID"]
    transitions = 0
    for i in range(1, len(filtered)):
        if filtered[i] != filtered[i - 1]:
            transitions += 1

    shows_switching = (
        transitions >= MIN_TRANSITIONS
        and pct_high >= MIN_REGIME_PCT
        and pct_low >= MIN_REGIME_PCT
    )

    return {
        "rolling": rolling,
        "transitions": transitions,
        "pct_high": pct_high,
        "pct_low": pct_low,
        "shows_switching": shows_switching,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/regime_detection.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_dir = data_dir / CSV_SUBDIR
    if not csv_dir.exists():
        _out(f"WARNING: KLiCKe CSV directory not found at {csv_dir}; skipping analysis.")
        json.dump({"n_writers_analyzed": 0, "n_writers_with_switching": 0,
                   "pct_with_switching": 0.0, "mean_transitions": 0.0,
                   "mean_high_pct": 0.0, "mean_low_pct": 0.0,
                   "example_profile": {}, "window_size": WINDOW, "stride": STRIDE},
                  open(out_path, "w"), indent=2)
        return

    csv_files = sorted(csv_dir.glob("*.csv"))
    _out(f"Found {len(csv_files)} KLiCKe writer CSVs")

    # Pass 1: compute global entropy thresholds across all writers
    _out("Pass 1: computing global entropy thresholds...")
    all_rolling: list[float] = []
    iki_cache: dict[str, list[float]] = {}
    for csv_path in csv_files:
        iki = _load_iki(csv_path)
        if len(iki) < MIN_IKI_VALUES:
            continue
        iki_cache[csv_path.stem] = iki
        rolling = _rolling_entropy(iki, WINDOW, STRIDE)
        all_rolling.extend(rolling)

    if not all_rolling:
        _out("No valid writers found.")
        return
    sorted_global = sorted(all_rolling)
    high_thresh = _percentile(sorted_global, 75.0)
    low_thresh = _percentile(sorted_global, 25.0)
    _out(f"  Global thresholds: HIGH >= {high_thresh:.3f}b, LOW <= {low_thresh:.3f}b")

    # Pass 2: detect regime switching using global thresholds
    n_analyzed = 0
    switchers: list[dict] = []
    all_transitions: list[int] = []
    all_high_pct: list[float] = []
    all_low_pct: list[float] = []
    example_profile: dict | None = None

    for writer_id, iki in sorted(iki_cache.items()):
        n_analyzed += 1
        csv_path = csv_dir / f"{writer_id}.csv"

        profile = _analyze_writer(iki, high_thresh, low_thresh)
        if profile is None:
            continue

        if profile["shows_switching"]:
            switchers.append({"writer_id": writer_id, **profile})
            all_transitions.append(profile["transitions"])
            all_high_pct.append(profile["pct_high"])
            all_low_pct.append(profile["pct_low"])

        if example_profile is None and profile["shows_switching"]:
            example_profile = {
                "writer_id": writer_id,
                "rolling_entropy": [round(v, 4) for v in profile["rolling"][:200]],
            }

        if n_analyzed % 500 == 0:
            _out(f"  Processed {n_analyzed} writers ({len(switchers)} with switching)...")

    n_switching = len(switchers)
    pct_switching = n_switching / n_analyzed if n_analyzed > 0 else 0.0
    mean_transitions = _mean([float(t) for t in all_transitions])
    mean_high_pct = _mean(all_high_pct)
    mean_low_pct = _mean(all_low_pct)

    _out(f"\nWriters analyzed        : {n_analyzed}")
    _out(f"Writers with switching  : {n_switching} ({pct_switching:.1%})")
    _out(f"Mean transitions        : {mean_transitions:.2f}")
    _out(f"Mean time HIGH (comp.)  : {mean_high_pct:.1%}")
    _out(f"Mean time LOW  (trans.) : {mean_low_pct:.1%}")
    if example_profile:
        _out(f"Example writer          : {example_profile['writer_id']} ({len(example_profile['rolling_entropy'])} windows shown)")

    result = {
        "n_writers_analyzed": n_analyzed,
        "n_writers_with_switching": n_switching,
        "pct_with_switching": round(pct_switching, 4),
        "mean_transitions": round(mean_transitions, 4),
        "mean_high_pct": round(mean_high_pct, 4),
        "mean_low_pct": round(mean_low_pct, 4),
        "example_profile": example_profile or {},
        "window_size": WINDOW,
        "stride": STRIDE,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
