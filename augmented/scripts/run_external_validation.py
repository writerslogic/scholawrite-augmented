"""Run cross-dataset validation of the embodied simulation.

Reports per-dataset comparisons and pooled statistics separately,
so each dataset's contribution is visible and not masked by N differences.

Usage:
    uv run python scripts/run_external_validation.py
    uv run python scripts/run_external_validation.py --task-type composition
    uv run python scripts/run_external_validation.py --data-dir data/external -o results/external_validation.json
    uv run python scripts/run_external_validation.py --dataset klicke --dataset cmu_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _out(*args: object, **kwargs: object) -> None:
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset validation of embodied simulation.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "external",
    )
    parser.add_argument("--dataset", action="append", help="Specific dataset(s) to validate against")
    parser.add_argument(
        "--task-type",
        choices=["composition", "transcription", "password", "mixed", "all"],
        default="all",
        help="Filter datasets by task type",
    )
    parser.add_argument("-n", "--n-sim-traces", type=int, default=200)
    parser.add_argument("--n-events", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, help="JSON output path")
    parser.add_argument("--quick", action="store_true", help="Reduced run (n=30)")
    args = parser.parse_args()

    if args.quick:
        args.n_sim_traces = 30
        args.n_events = 30

    from scholawrite.datasets import REGISTRY, TaskType, load_all, load_by_task_type, load_dataset
    from scholawrite.validation import compare_distributions, generate_simulation_reference

    _out("Generating simulation reference traces...")
    sim_checkpoints = generate_simulation_reference(
        n_traces=args.n_sim_traces,
        n_events=args.n_events,
        seed=args.seed,
    )
    _out(f"  {len(sim_checkpoints)} simulation checkpoints generated.\n")

    if args.dataset:
        datasets = {}
        for name in args.dataset:
            try:
                records = load_dataset(name, args.data_dir)
                if records:
                    datasets[name] = records
            except (FileNotFoundError, KeyError) as e:
                _out(f"  Skipping {name}: {e}")
    elif args.task_type != "all":
        task_map = {
            "composition": [TaskType.COMPOSITION],
            "transcription": [TaskType.TRANSCRIPTION],
            "password": [TaskType.PASSWORD],
            "mixed": [TaskType.MIXED],
        }
        _out(f"Loading datasets with task_type={args.task_type}...")
        datasets = load_by_task_type(args.data_dir, task_map[args.task_type])
    else:
        _out("Loading all available datasets...")
        datasets = load_all(args.data_dir)

    if not datasets:
        _out("No datasets found. Run fetch_datasets.py first.")
        sys.exit(1)

    _out(f"Loaded {len(datasets)} dataset(s).\n")

    signals = ["mean_iki_ms", "iki_entropy_bits", "lag1_autocorrelation", "revision_density", "wpm"]
    all_results = {}

    for ds_name, records in sorted(datasets.items()):
        meta = REGISTRY[ds_name].METADATA
        _out(f"=== {meta.name} (n={len(records)}, task={meta.task_type.value}) ===")

        ds_results = {"n_checkpoints": len(records), "task_type": meta.task_type.value, "signals": {}}

        for signal in signals:
            sim_vals = [getattr(cp, signal) for cp in sim_checkpoints if getattr(cp, signal) is not None]
            real_vals = [getattr(cp, signal) for cp in records if getattr(cp, signal) is not None]
            if len(sim_vals) < 5 or len(real_vals) < 5:
                continue
            comparison = compare_distributions(real_vals, sim_vals, signal)
            ds_results["signals"][signal] = asdict(comparison)
            _out(f"    {signal}: KS={comparison.ks_statistic:.3f} p={comparison.ks_pvalue:.4f}  d={comparison.cohens_d:.3f}  W={comparison.wasserstein_distance:.3f}")

        all_results[ds_name] = ds_results
        _out("")

    _out(f"\n{'Dataset':<25} {'N':>6} {'Task':<15} {'IKI KS':>8} {'Ent KS':>8} {'d':>7}")
    _out("-" * 75)
    for ds_name, res in sorted(all_results.items()):
        sigs = res["signals"]
        iki_ks = sigs.get("mean_iki_ms", {}).get("ks_statistic")
        ent_ks = sigs.get("iki_entropy_bits", {}).get("ks_statistic")
        cohens_d = sigs.get("mean_iki_ms", {}).get("cohens_d")
        _out(
            f"{ds_name:<25} {res['n_checkpoints']:>6} {res['task_type']:<15}"
            f" {iki_ks:>8.3f}" if iki_ks is not None else f"{'n/a':>8}",
            f" {ent_ks:>8.3f}" if ent_ks is not None else f"{'n/a':>8}",
            f" {cohens_d:>7.3f}" if cohens_d is not None else f"{'n/a':>7}",
        )

    _out("\n--- Pooled (all datasets combined) ---")
    all_records = [cp for records in datasets.values() for cp in records]
    _out(f"Total checkpoints: {len(all_records)}")
    for signal in signals:
        sim_vals = [getattr(cp, signal) for cp in sim_checkpoints if getattr(cp, signal) is not None]
        real_vals = [getattr(cp, signal) for cp in all_records if getattr(cp, signal) is not None]
        if len(sim_vals) < 5 or len(real_vals) < 5:
            continue
        comparison = compare_distributions(real_vals, sim_vals, signal)
        _out(f"  {signal}: KS={comparison.ks_statistic:.3f} p={comparison.ks_pvalue:.4f}  d={comparison.cohens_d:.3f}  W={comparison.wasserstein_distance:.3f}")
        all_results["_pooled"] = all_results.get("_pooled", {"n_checkpoints": len(all_records), "signals": {}})
        all_results["_pooled"]["signals"][signal] = asdict(comparison)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        _out(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
