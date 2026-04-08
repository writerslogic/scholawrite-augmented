"""Models keystroke detection as an information channel and computes channel capacity
per adversarial tier.

Mutual information I(signal; label) quantifies how much cognitive information survives
forgery. Shows capacity degrades with adversarial knowledge.

Paper relevance: connects empirical adversarial results to Shannon information theory,
providing principled upper bounds.

Usage:
    uv run python analysis/extended/10_channel_capacity.py --data-dir data/ -o results/channel_capacity.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import entropy_bits, lag1_autocorr, load_all, REGISTRY, TaskType
from scholawrite.validation import generate_simulation_reference


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _h_binary(p: float) -> float:
    """Binary entropy function H(p) in bits. Returns 0 at boundaries."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _bsc_capacity(auc: float) -> float:
    """Binary-symmetric channel capacity approximation from AUC.

    Treats AUC as P(correct detection) in a binary hypothesis test.
    C = 1 - H_binary(p_detect) gives the channel capacity in bits.
    AUC = 0.5 (chance) yields C = 0; AUC = 1.0 yields C = 1.
    """
    p = max(0.0, min(1.0, auc))
    return round(1.0 - _h_binary(p), 6)


# Ordered from weakest (naive) to strongest (Markov-aware) adversary.
TIER_ORDER = ["attack_constant", "attack_iid", "attack_cross", "attack_markov"]
TIER_LABELS = ["constant (naive)", "iid (statistical)", "cross-user", "markov"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory (unused; included for interface consistency)")
    parser.add_argument("-o", "--output", default="results/channel_capacity.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_path = Path(__file__).parent.parent / "retype_simulation_results.json"
    if not results_path.exists():
        _out(f"WARNING: {results_path} not found. Run analysis/retype_simulation.py first.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tiers": TIER_LABELS, "per_signal_capacity_bits": {},
                   "total_capacity_per_tier": [], "capacity_degradation_ratio": [],
                   "tier_where_capacity_below_half": None}, open(out_path, "w"), indent=2)
        return

    try:
        with open(results_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _out(f"WARNING: could not load retype results: {exc}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tiers": TIER_LABELS, "per_signal_capacity_bits": {},
                   "total_capacity_per_tier": [], "capacity_degradation_ratio": [],
                   "tier_where_capacity_below_half": None}, open(out_path, "w"), indent=2)
        return

    dba = data.get("detection_by_attack_type", {})

    # Collect all signals across tiers.
    all_signals: set[str] = set()
    for tier in TIER_ORDER:
        all_signals.update(dba.get(tier, {}).keys())
    signals = sorted(all_signals)

    _out(f"Signals found: {signals}")
    _out(f"Tiers: {TIER_LABELS}")
    _out("")

    # Compute per-signal capacity across tiers.
    per_signal_capacity: dict[str, list[float]] = {}
    for sig in signals:
        caps = []
        for tier in TIER_ORDER:
            tier_data = dba.get(tier, {})
            sig_data = tier_data.get(sig, {})
            auc = sig_data.get("auc_mann_whitney", 0.5)
            caps.append(_bsc_capacity(auc))
        per_signal_capacity[sig] = caps

    # Total capacity per tier = sum over signals.
    total_capacity: list[float] = []
    for t_idx in range(len(TIER_ORDER)):
        total = sum(per_signal_capacity[sig][t_idx] for sig in signals)
        total_capacity.append(round(total, 6))

    # Degradation ratio relative to tier-0 (naive adversary).
    c0 = total_capacity[0]
    if c0 > 0.0:
        degradation = [round(c / c0, 6) for c in total_capacity]
    else:
        degradation = [1.0 if i == 0 else 0.0 for i in range(len(TIER_ORDER))]

    # First tier where total capacity drops below half of tier-0.
    half_threshold = c0 / 2.0
    below_half: int | None = None
    for i, c in enumerate(total_capacity):
        if c < half_threshold:
            below_half = i
            break

    # Print summary table.
    _out(f"{'Tier':<28}  {'Total C (bits)':>16}  {'C/C0':>8}  {'Signals contributing':>6}")
    _out("-" * 70)
    for i, (tier, label) in enumerate(zip(TIER_ORDER, TIER_LABELS)):
        n_contributing = sum(1 for sig in signals if per_signal_capacity[sig][i] > 0.01)
        _out(f"  {label:<26}  {total_capacity[i]:>16.4f}  {degradation[i]:>8.4f}  {n_contributing:>6}")

    _out("")
    if below_half is not None:
        _out(f"Capacity drops below 50% of naive at tier index {below_half} ({TIER_LABELS[below_half]})")
    else:
        _out("Capacity remains above 50% of naive across all tiers")

    _out("\nPer-signal capacity by tier:")
    header = f"  {'Signal':<20}" + "".join(f"  {lbl[:10]:>12}" for lbl in TIER_LABELS)
    _out(header)
    for sig in signals:
        row = f"  {sig:<20}" + "".join(f"  {v:>12.4f}" for v in per_signal_capacity[sig])
        _out(row)

    result = {
        "tiers": TIER_LABELS,
        "per_signal_capacity_bits": per_signal_capacity,
        "total_capacity_per_tier": total_capacity,
        "capacity_degradation_ratio": degradation,
        "tier_where_capacity_below_half": below_half,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
