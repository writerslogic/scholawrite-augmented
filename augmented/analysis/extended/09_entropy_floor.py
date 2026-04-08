"""Tests whether human cognition has a minimum IKI entropy floor that AI cannot replicate.

Computes per-dataset entropy percentiles (1st, 5th, 10th) and compares against
simulation's entropy distribution. If the human 5th percentile is above the
simulation's 95th percentile, the floor claim holds.

Paper relevance: strongest possible evidence that process signals capture something
fundamental about human cognition.

Usage:
    uv run python analysis/extended/09_entropy_floor.py --data-dir data/ -o results/entropy_floor.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import REGISTRY, TaskType, entropy_bits, lag1_autocorr, load_all
from scholawrite.validation import generate_simulation_reference


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted sequence."""
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


def _percentile_map(sorted_vals: list[float], percentiles: list[int]) -> dict[str, float]:
    return {f"p{p}": round(_percentile(sorted_vals, float(p)), 6) for p in percentiles}


PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# Exclude task types that do not reflect natural human composition cognition.
EXCLUDED_TASK_TYPES = {TaskType.REFERENCE.value, TaskType.PASSWORD.value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/entropy_floor.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _out("Loading all datasets...")
    all_datasets = load_all(data_dir)
    _out(f"  Loaded {len(all_datasets)} datasets with data")

    per_dataset: dict[str, dict] = {}
    pooled_human: list[float] = []

    for ds_name, records in all_datasets.items():
        loader = REGISTRY.get(ds_name)
        if loader is None:
            continue
        task_type_val = loader.METADATA.task_type.value
        exclude = task_type_val in EXCLUDED_TASK_TYPES

        vals = sorted(
            r.iki_entropy_bits for r in records if r.iki_entropy_bits > 0.0
        )
        if not vals:
            continue

        pmap = _percentile_map(vals, PERCENTILES)
        pmap["n"] = len(vals)
        per_dataset[ds_name] = pmap

        if not exclude:
            pooled_human.extend(vals)

    _out(f"  Per-dataset entropy profiles built for {len(per_dataset)} datasets")
    _out(f"  Pooled human (non-reference, non-password): {len(pooled_human)} checkpoints")

    # Simulation reference
    _out("\nGenerating simulation reference (200 traces, 100 events each)...")
    sim_traces = generate_simulation_reference(200, 100, 42)
    sim_entropy = sorted(
        t["iki_entropy_bits"] for t in sim_traces if t.get("iki_entropy_bits", 0.0) > 0.0
    )
    _out(f"  Simulation entropy values: {len(sim_entropy)}")

    pooled_human_sorted = sorted(pooled_human)

    # Compute summary percentile maps
    human_pmap = _percentile_map(pooled_human_sorted, [1, 5, 10, 50, 95])
    human_pmap["n"] = len(pooled_human_sorted)

    sim_pmap = _percentile_map(sim_entropy, [1, 5, 10, 50, 95])
    sim_pmap["n"] = len(sim_entropy)

    # Floor claim evaluation
    human_p5 = human_pmap["p5"]
    sim_p95 = sim_pmap["p95"]
    gap_bits = round(human_p5 - sim_p95, 6)
    floor_holds = bool(human_p5 > sim_p95)

    _out("\n--- Floor Claim ---")
    _out(f"  Human p5 entropy : {human_p5:.4f} bits")
    _out(f"  Simulation p95   : {sim_p95:.4f} bits")
    _out(f"  Gap              : {gap_bits:+.4f} bits")
    _out(f"  Floor holds      : {floor_holds}")

    if floor_holds:
        _out("  RESULT: The entropy floor claim holds. Human writers have")
        _out("          a higher 5th-percentile entropy than the simulation's")
        _out("          95th percentile. This supports the process attestation thesis.")
    else:
        _out("  RESULT: Floor claim does NOT hold. Distributions overlap at this")
        _out("          threshold. Consider tightening dataset filters or examining")
        _out("          per-dataset profiles for confounders.")

    _out("\nPer-dataset entropy summary (p5 / p50 / p95):")
    for ds, pmap in sorted(per_dataset.items()):
        _out(f"  {ds:30s}  p5={pmap['p5']:.3f}  p50={pmap['p50']:.3f}  p95={pmap['p95']:.3f}  n={pmap['n']}")

    result = {
        "per_dataset_percentiles": per_dataset,
        "pooled_human": human_pmap,
        "simulation": sim_pmap,
        "floor_claim": {
            "holds": floor_holds,
            "human_p5": human_p5,
            "sim_p95": sim_p95,
            "gap_bits": gap_bits,
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
