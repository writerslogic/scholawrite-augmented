"""Quantifies the computational cost of forgery across adversarial tiers.

Shows AUC degradation per tier and marginal cost per upgrade. Uses pre-computed
retype simulation and adversarial evaluation results. Paper relevance: empirically
validates the VDF analogy (sequential > parallel forgery hardness).

Attack tiers in order of sophistication:
  constant  -- fixed-rate replay (zero knowledge)
  iid       -- distribution-matched iid sampling (first-order stats matched)
  cross     -- cross-writer transfer (population-level structure)
  markov    -- Markov-chain retype preserving lag-1 autocorrelation

For each signal the AUC at each tier is read from retype_simulation_results.json.
Marginal forgery cost is the AUC drop when upgrading from one tier to the next.
Detection collapse is the first tier where AUC < 0.55.

Usage:
    uv run python analysis/extended/07_forgery_cost_analysis.py \\
        --data-dir /path/to/augmented -o results/07_forgery_cost_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


TIERS = ["constant", "iid", "cross", "markov"]
TIER_KEYS = ["attack_constant", "attack_iid", "attack_cross", "attack_markov"]
NEAR_RANDOM_THRESHOLD = 0.55


def _load_retype_results(data_dir: Path) -> dict | None:
    path = data_dir / "analysis" / "retype_simulation_results.json"
    if not path.exists():
        _out(f"WARNING: retype_simulation_results.json not found at {path}")
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _out(f"WARNING: could not load retype results: {exc}")
        return None


def _load_ablation_results(data_dir: Path) -> Optional[dict]:
    path = data_dir / "ablation_results.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Path to augmented/ directory (contains analysis/ and ablation_results.json)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("07_forgery_cost_analysis.json"))
    args = parser.parse_args()

    retype = _load_retype_results(args.data_dir)
    if retype is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tiers": TIERS, "per_signal_auc": {}, "composite_auc_per_tier": [],
                   "marginal_cost": {}, "detection_collapses_at_tier": {},
                   "near_random_threshold": NEAR_RANDOM_THRESHOLD, "ablation_available": False}, open(args.output, "w"), indent=2)
        return
    ablation = _load_ablation_results(args.data_dir)

    detection = retype.get("detection_by_attack_type", {})

    # Collect all signal names present across all tiers
    signal_names: List[str] = []
    for tier_key in TIER_KEYS:
        for sig in detection.get(tier_key, {}):
            if sig not in signal_names:
                signal_names.append(sig)

    _out(f"Signals found: {signal_names}")
    _out(f"Tiers: {TIERS}")

    per_signal_auc: Dict[str, List[Optional[float]]] = {}
    for sig in signal_names:
        aucs: List[Optional[float]] = []
        for tier_key in TIER_KEYS:
            tier_data = detection.get(tier_key, {}).get(sig, {})
            auc_val = tier_data.get("auc_mann_whitney")
            aucs.append(round(auc_val, 4) if auc_val is not None else None)
        per_signal_auc[sig] = aucs

    # Composite AUC per tier: mean across signals with available values
    composite_per_tier: List[Optional[float]] = []
    for t_idx in range(len(TIERS)):
        vals = [
            per_signal_auc[sig][t_idx]
            for sig in signal_names
            if per_signal_auc[sig][t_idx] is not None
        ]
        composite_per_tier.append(round(sum(vals) / len(vals), 4) if vals else None)

    # Marginal cost: delta AUC between consecutive tiers (negative = degradation)
    marginal_cost: Dict[str, List[Optional[float]]] = {}
    for sig in signal_names:
        aucs = per_signal_auc[sig]
        deltas: List[Optional[float]] = []
        for t_idx in range(1, len(TIERS)):
            prev = aucs[t_idx - 1]
            curr = aucs[t_idx]
            if prev is not None and curr is not None:
                deltas.append(round(curr - prev, 4))
            else:
                deltas.append(None)
        marginal_cost[sig] = deltas

    # Detection collapse: first tier index where AUC < threshold
    detection_collapses_at_tier: Dict[str, Optional[int]] = {}
    for sig in signal_names:
        aucs = per_signal_auc[sig]
        collapse: Optional[int] = None
        for t_idx, auc_val in enumerate(aucs):
            if auc_val is not None and auc_val < NEAR_RANDOM_THRESHOLD:
                collapse = t_idx
                break
        detection_collapses_at_tier[sig] = collapse

    _out("\nPer-signal AUC across tiers:")
    header = f"  {'signal':20s}" + "".join(f"  {t:10s}" for t in TIERS)
    _out(header)
    for sig in signal_names:
        row = f"  {sig:20s}"
        for auc_val in per_signal_auc[sig]:
            row += f"  {auc_val!s:10s}" if auc_val is None else f"  {auc_val:.4f}    "
        _out(row)

    _out("\nComposite AUC per tier:")
    for t_idx, (tier, comp) in enumerate(zip(TIERS, composite_per_tier)):
        _out(f"  {tier:12s}: {comp}")

    _out("\nDetection collapse tier:")
    for sig, col in detection_collapses_at_tier.items():
        tier_name = TIERS[col] if col is not None else "never"
        _out(f"  {sig:20s}: {tier_name}")

    output = {
        "tiers": TIERS,
        "per_signal_auc": per_signal_auc,
        "composite_auc_per_tier": composite_per_tier,
        "marginal_cost": marginal_cost,
        "detection_collapses_at_tier": detection_collapses_at_tier,
        "near_random_threshold": NEAR_RANDOM_THRESHOLD,
        "ablation_available": ablation is not None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
    _out(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
