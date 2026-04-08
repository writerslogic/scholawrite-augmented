"""Tests whether process signals form a biometric fingerprint.

Uses 1-nearest-neighbor identification on KLiCKe writers. If rank-1 accuracy
is well above chance (1/N), signals are biometric; if near chance, they are
universal cognitive properties. Paper relevance: distinguishes individual from
universal cognitive signatures.

Each writer's keystroke log is split into a first-half probe and second-half
gallery entry. Cross-half 1-NN identification is then performed: for writer i,
find the gallery entry whose feature vector is nearest to the probe. Rank-1
accuracy, rank-5 accuracy, rank-10 accuracy, and identification entropy are
reported against their respective chance baselines.

Usage:
    uv run python analysis/extended/06_writer_identification.py \\
        --data-dir /path/to/data -o results/06_writer_identification.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


IKI_MIN_MS = 10.0
IKI_MAX_MS = 60_000.0
MIN_EVENTS = 20


def _feats_from_rows(rows: List[dict]) -> Optional[List[float]]:
    iki_ms: List[float] = []
    input_chars = 0
    remove_events = 0
    nonprod_events = 0
    total_events = 0
    prev_down: Optional[float] = None
    first_down: Optional[float] = None
    last_down: Optional[float] = None

    for row in rows:
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

    if total_events < MIN_EVENTS // 2 or len(iki_ms) < MIN_EVENTS // 2:
        return None

    mean_iki = sum(iki_ms) / len(iki_ms)
    variance = sum((x - mean_iki) ** 2 for x in iki_ms) / len(iki_ms)
    std_iki = math.sqrt(variance)
    ent = entropy_bits(iki_ms)
    ac = lag1_autocorr(iki_ms) or 0.0
    revision_density = remove_events / total_events if total_events > 0 else 0.0
    planning_ratio = nonprod_events / total_events if total_events > 0 else 0.0
    total_time_ms = (last_down - first_down) if (last_down and first_down) else sum(iki_ms)
    wpm = (input_chars / 5.0) / (total_time_ms / 60_000.0) if total_time_ms > 0 else 0.0

    return [mean_iki, std_iki, ent, ac, revision_density, planning_ratio, wpm]


def _global_mean_std(matrix: List[List[float]]) -> Tuple[List[float], List[float]]:
    n = len(matrix)
    d = len(matrix[0])
    means = [sum(row[j] for row in matrix) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in matrix) / n
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return means, stds


def _normalize(vec: List[float], means: List[float], stds: List[float]) -> List[float]:
    return [(vec[j] - means[j]) / stds[j] for j in range(len(vec))]


def _sq_dist(a: List[float], b: List[float]) -> float:
    return sum((a[j] - b[j]) ** 2 for j in range(len(a)))


def _identification_entropy(pos_counts: Dict[int, int]) -> float:
    total = sum(pos_counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in pos_counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("-o", "--output", type=Path, default=Path("06_writer_identification.json"))
    args = parser.parse_args()

    csv_dir = (
        args.data_dir
        / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
    )
    if not csv_dir.exists():
        _out(f"WARNING: KLiCKe CSV dir not found: {csv_dir}; skipping analysis.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_writers": 0, "rank1_accuracy": None, "rank5_accuracy": None, "rank10_accuracy": None,
                   "chance_rank1": None, "chance_rank5": None, "chance_rank10": None,
                   "identification_entropy_bits": None, "above_chance_ratio": None}, open(args.output, "w"), indent=2)
        return

    csv_files = sorted(csv_dir.glob("*.csv"))
    _out(f"Processing {len(csv_files)} writer CSV files...")

    half_a: List[List[float]] = []
    half_b: List[List[float]] = []
    skipped = 0

    for i, csv_path in enumerate(csv_files):
        if (i + 1) % 500 == 0:
            _out(f"  {i+1}/{len(csv_files)}...")
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
                all_rows = list(csv.DictReader(fh))
        except Exception:
            skipped += 1
            continue

        if len(all_rows) < MIN_EVENTS * 2:
            skipped += 1
            continue

        mid = len(all_rows) // 2
        va = _feats_from_rows(all_rows[:mid])
        vb = _feats_from_rows(all_rows[mid:])
        if va is None or vb is None:
            skipped += 1
            continue

        half_a.append(va)
        half_b.append(vb)

    n = len(half_a)
    _out(f"  Usable writers: {n}  (skipped {skipped})")
    if n < 10:
        _out(f"WARNING: Too few writers for split evaluation ({n}); skipping analysis.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_writers": n, "rank1_accuracy": None, "rank5_accuracy": None, "rank10_accuracy": None,
                   "chance_rank1": None, "chance_rank5": None, "chance_rank10": None,
                   "identification_entropy_bits": None, "above_chance_ratio": None}, open(args.output, "w"), indent=2)
        return

    means, stds = _global_mean_std(half_b)
    norm_a = [_normalize(v, means, stds) for v in half_a]
    norm_b = [_normalize(v, means, stds) for v in half_b]

    _out("Running cross-half 1-NN identification...")

    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0
    pos_counts: Dict[int, int] = {}

    for i in range(n):
        probe = norm_a[i]
        dists = sorted(
            ((math.sqrt(_sq_dist(probe, norm_b[j])), j) for j in range(n)),
            key=lambda t: t[0],
        )
        pos = next(k for k, (_, j) in enumerate(dists) if j == i)
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        if pos == 0:
            rank1_correct += 1
        if pos < 5:
            rank5_correct += 1
        if pos < 10:
            rank10_correct += 1

    rank1_acc = rank1_correct / n
    rank5_acc = rank5_correct / n
    rank10_acc = rank10_correct / n
    chance1 = 1.0 / n
    chance5 = min(1.0, 5.0 / n)
    chance10 = min(1.0, 10.0 / n)
    above_chance = rank1_acc / chance1
    ident_entropy = _identification_entropy(pos_counts)

    _out(f"\nResults (n={n} writers):")
    _out(f"  Rank-1  accuracy: {rank1_acc:.4f}  (chance={chance1:.6f}, ratio={above_chance:.2f}x)")
    _out(f"  Rank-5  accuracy: {rank5_acc:.4f}  (chance={chance5:.6f})")
    _out(f"  Rank-10 accuracy: {rank10_acc:.4f}  (chance={chance10:.6f})")
    _out(f"  Identification entropy: {ident_entropy:.3f} bits")

    output = {
        "n_writers": n,
        "rank1_accuracy": round(rank1_acc, 4),
        "rank5_accuracy": round(rank5_acc, 4),
        "rank10_accuracy": round(rank10_acc, 4),
        "chance_rank1": round(chance1, 6),
        "chance_rank5": round(chance5, 6),
        "chance_rank10": round(chance10, 6),
        "identification_entropy_bits": round(ident_entropy, 4),
        "above_chance_ratio": round(above_chance, 3),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
    _out(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
