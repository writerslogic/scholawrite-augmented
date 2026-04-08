"""CLI script for the cross-author universality experiment.

Runs the full leave-one-out experiment and author similarity analysis,
then outputs JSON results and a printed summary table.
"""
from __future__ import annotations

import argparse
import json
import sys

from scholawrite.cross_author import CrossAuthorExperiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-author universality test for causal signatures"
    )
    parser.add_argument("--n-authors", type=int, default=5)
    parser.add_argument("--traces-per-author", type=int, default=50)
    parser.add_argument("--tokens-per-trace", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    exp = CrossAuthorExperiment(
        n_authors=args.n_authors,
        traces_per_author=args.traces_per_author,
        tokens_per_trace=args.tokens_per_trace,
        seed=args.seed,
    )

    print("=" * 60)
    print("CROSS-AUTHOR UNIVERSALITY EXPERIMENT")
    print("=" * 60)
    print(f"Authors: {args.n_authors}")
    print(f"Traces per author: {args.traces_per_author}")
    print(f"Tokens per trace: {args.tokens_per_trace}")
    print(f"Seed: {args.seed}")
    print()

    # Leave-one-out
    print("Running leave-one-out cross-validation...")
    loo_results = exp.run_leave_one_out()

    print()
    print("-" * 60)
    print("LEAVE-ONE-OUT RESULTS")
    print("-" * 60)
    print(f"{'Fold':<12} {'Author':<14} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    for i, fold in enumerate(loo_results["fold_results"]):
        print(
            f"{i:<12} {fold['test_author']:<14} "
            f"{fold['accuracy']:<10.4f} {fold['precision']:<10.4f} "
            f"{fold['recall']:<10.4f} {fold['f1']:<10.4f}"
        )
    print("-" * 60)
    print(f"{'Mean F1:':<26} {loo_results['mean_f1']:.4f} (+/- {loo_results['std_f1']:.4f})")
    print(f"{'Min F1:':<26} {loo_results['min_f1']:.4f}")
    print(f"{'Max F1:':<26} {loo_results['max_f1']:.4f}")
    print(f"{'Mean Accuracy:':<26} {loo_results['mean_accuracy']:.4f}")
    print()

    # Author similarity
    print("Running author similarity analysis...")
    sim_results = exp.run_author_similarity_analysis()

    print()
    print("-" * 60)
    print("AUTHOR SIMILARITY ANALYSIS")
    print("-" * 60)
    print(f"Mean pairwise similarity: {sim_results['mean_similarity']:.4f}")
    print(f"Min pairwise similarity:  {sim_results['min_similarity']:.4f}")
    print()
    print("Universality ranking (most to least universal):")
    for rank, metric in enumerate(sim_results["universality_ranking"], 1):
        std_val = sim_results["per_metric_std"][metric]
        print(f"  {rank}. {metric}: std={std_val:.6f}")

    print()
    print("=" * 60)
    verdict = "YES" if loo_results["universal"] else "NO"
    print(f"Are causal signatures universal? {verdict}")
    if loo_results["universal"]:
        print("  Evidence: All fold F1 scores > 0.7, indicating thresholds")
        print("  trained on one set of authors transfer to unseen authors.")
    else:
        print(f"  Evidence: Min fold F1 = {loo_results['min_f1']:.4f} (threshold: 0.7)")
        print("  Some folds show poor transfer, suggesting author-specific")
        print("  variation in signatures.")
    print("=" * 60)

    # Combine results
    combined = {
        "leave_one_out": loo_results,
        "author_similarity": sim_results,
        "verdict": {
            "universal": loo_results["universal"],
            "mean_f1": loo_results["mean_f1"],
            "min_f1": loo_results["min_f1"],
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        # Print JSON to stdout
        print("\n--- JSON Results ---")
        print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
