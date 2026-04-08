#!/usr/bin/env python3
"""Run the gradient-based forgery convergence study.

This is the empirical complement to the impossibility theorem. It shows
whether an optimizer can efficiently find traces that match human signal
distributions, and how quickly it converges.

Usage:
    # Quick test
    uv run python scripts/run_gradient_forger.py --n-runs 5 --max-iter 100

    # Full study for paper
    uv run python scripts/run_gradient_forger.py --n-runs 20 --max-iter 500 -o results/convergence.json

    # With LaTeX table
    uv run python scripts/run_gradient_forger.py --n-runs 20 --max-iter 500 --output-latex results/convergence.tex
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, stdev
from typing import Any, Dict, List

from scholawrite.gradient_forger import GradientForger, EvolutionaryForger


def _fmt_ci(values: List[float]) -> str:
    """Format mean with 95% CI from a list of values."""
    if len(values) < 2:
        return f"{mean(values):.4f}" if values else "N/A"
    m = mean(values)
    s = stdev(values)
    n = len(values)
    # 95% CI via t-approx (1.96 for large n)
    half = 1.96 * s / (n ** 0.5)
    return f"{m:.4f} ({m - half:.4f}-{m + half:.4f})"


def _write_latex_table(
    gradient_results: Dict[str, Any],
    evo_results: Dict[str, Any],
    path: str,
) -> None:
    """Write a booktabs comparison table for gradient vs evolutionary forger."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Forgery convergence: gradient descent vs.\ evolutionary strategy.}",
        r"\label{tab:forger-convergence}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Gradient & Evolutionary \\",
        r"\midrule",
    ]

    metrics = [
        ("Convergence rate", "convergence_rate", ".1%"),
        ("Mean initial loss", "mean_initial_loss", ".6f"),
        ("Mean final loss", "mean_final_loss", ".6f"),
        ("Loss reduction ratio", "loss_reduction_ratio", ".4f"),
    ]
    for label, key, fmt in metrics:
        g_val = gradient_results.get(key, 0.0)
        e_val = evo_results.get(key, 0.0)
        g_str = f"{g_val:{fmt}}"
        e_str = f"{e_val:{fmt}}"
        lines.append(rf"{label} & {g_str} & {e_str} \\")

    # Per-signal deviation sub-table
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{\textit{Per-signal mean deviation from target}} \\")
    lines.append(r"\midrule")

    g_devs = gradient_results.get("per_signal_mean_deviation", {})
    e_devs = evo_results.get("per_signal_mean_deviation", {})
    for sig in g_devs:
        sig_label = sig.replace("_", r"\_")
        g_val = g_devs.get(sig, 0.0)
        e_val = e_devs.get(sig, 0.0)
        lines.append(rf"{sig_label} & {g_val:.4f} & {e_val:.4f} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient forger convergence study")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--output-latex", type=str, default=None, help="Path for booktabs LaTeX table")
    args = parser.parse_args()

    print("\n=== Gradient Forger Convergence Study ===\n")
    print(f"  Runs:           {args.n_runs}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Trace length:   {args.n_events} events")
    print(f"  Learning rate:  {args.lr}")
    print()

    forger = GradientForger(
        n_events=args.n_events,
        seed=args.seed,
        learning_rate=args.lr,
    )

    # Compute human targets
    print("  Computing human target signals...")
    targets = forger.compute_human_targets(n_traces=30)
    print("  Targets:")
    for k, v in targets.items():
        print(f"    {k:<25} {v:.4f}")
    print()

    # Run convergence study
    print("  Running convergence study...")
    results = forger.run_convergence_study(
        n_runs=args.n_runs,
        max_iterations=args.max_iter,
        targets=targets,
    )

    # Print results
    print(f"\n  --- Results ---\n")
    print(f"  Convergence rate:      {results['convergence_rate']:.1%}")
    print(f"  Mean initial loss:     {results['mean_initial_loss']:.6f}")
    print(f"  Mean final loss:       {results['mean_final_loss']:.6f}")
    print(f"  Loss reduction ratio:  {results['loss_reduction_ratio']:.4f}")
    if results['mean_convergence_iter']:
        print(f"  Mean convergence iter: {results['mean_convergence_iter']:.0f}")
    print()

    print("  Per-signal mean deviation from target:")
    for sig, dev in results["per_signal_mean_deviation"].items():
        quality = "matched" if dev < 0.05 else "close" if dev < 0.15 else "unmatched"
        print(f"    {sig:<25} {dev:.4f}  ({quality})")

    print()
    print(f"  Interpretation: {results['interpretation']}")
    print()

    # Run evolutionary strategy for comparison
    print("\n=== Evolutionary Forger Convergence Study ===\n")
    evo_forger = EvolutionaryForger(
        n_events=args.n_events,
        seed=args.seed,
        population_size=20,
        sigma=0.1,
    )
    evo_results = evo_forger.run_convergence_study(
        n_runs=args.n_runs,
        max_generations=args.max_iter,
        targets=targets,
    )

    print(f"  Convergence rate:      {evo_results['convergence_rate']:.1%}")
    print(f"  Mean initial loss:     {evo_results['mean_initial_loss']:.6f}")
    print(f"  Mean final loss:       {evo_results['mean_final_loss']:.6f}")
    print(f"  Loss reduction ratio:  {evo_results['loss_reduction_ratio']:.4f}")
    if evo_results["mean_convergence_iter"]:
        print(f"  Mean convergence gen:  {evo_results['mean_convergence_iter']:.0f}")
    print()

    print("  Per-signal mean deviation from target:")
    for sig, dev in evo_results["per_signal_mean_deviation"].items():
        quality = "matched" if dev < 0.05 else "close" if dev < 0.15 else "unmatched"
        print(f"    {sig:<25} {dev:.4f}  ({quality})")

    print()
    print(f"  Interpretation: {evo_results['interpretation']}")
    print()

    # Comparison summary
    print("=== Comparison: Gradient vs Evolutionary ===\n")
    print(f"  {'Method':<15} {'Conv Rate':>10} {'Loss Ratio':>12} {'Interpretation'}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*40}")
    print(f"  {'Gradient':<15} {results['convergence_rate']:>10.1%} {results['loss_reduction_ratio']:>12.4f} {results['interpretation'][:40]}")
    print(f"  {'Evolutionary':<15} {evo_results['convergence_rate']:>10.1%} {evo_results['loss_reduction_ratio']:>12.4f} {evo_results['interpretation'][:40]}")
    print()

    # Write LaTeX table
    if args.output_latex:
        _write_latex_table(results, evo_results, args.output_latex)
        print(f"  LaTeX table written to: {args.output_latex}")

    # Write JSON
    output = {
        "config": {
            "n_runs": args.n_runs,
            "max_iterations": args.max_iter,
            "n_events": args.n_events,
            "learning_rate": args.lr,
            "seed": args.seed,
        },
        "gradient": results,
        "evolutionary": evo_results,
    }
    report = json.dumps(output, indent=2)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"  Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
