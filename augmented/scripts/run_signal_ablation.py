#!/usr/bin/env python
"""Signal ablation study for consciousness-correlate discrimination.

Computes per-signal contribution to composite AUC across all 4 adversary tiers
using leave-one-out analysis, standalone evaluation, and bootstrap CIs.

Usage:
    uv run python scripts/run_signal_ablation.py -n 50
    uv run python scripts/run_signal_ablation.py -n 100 --output results.json --output-latex tables.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scholawrite.adversarial import AdversarialEvaluator, ALL_TIERS


def print_leave_one_out_table(results: Dict[str, Any]) -> None:
    """Print leave-one-out AUC table (tier x signal)."""
    signal_names = results["signal_names"]
    tier_results = results["tier_results"]

    print("\n" + "=" * 90)
    print("  LEAVE-ONE-OUT AUC  (composite AUC with signal removed, weights renormalized)")
    print("=" * 90)

    # Header
    tier_labels = {t: t[:7] for t in ALL_TIERS}
    header = f"{'Signal':<22}"
    for tier in ALL_TIERS:
        header += f" {'AUC w/o':>8} {'delta':>7}"
    print(header)
    print("-" * 90)

    for sig in signal_names:
        row = f"{sig:<22}"
        for tier in ALL_TIERS:
            loo = tier_results[tier]["leave_one_out"][sig]
            row += f" {loo['auc']:>8.4f} {loo['delta']:>+7.4f}"
        print(row)

    # Full composite row
    row = f"{'FULL COMPOSITE':<22}"
    for tier in ALL_TIERS:
        full = tier_results[tier]["full_composite"]
        row += f" {full['auc']:>8.4f} {'---':>7}"
    print("-" * 90)
    print(row)
    print()


def print_standalone_table(results: Dict[str, Any]) -> None:
    """Print standalone AUC table (tier x signal)."""
    signal_names = results["signal_names"]
    tier_results = results["tier_results"]

    print("=" * 82)
    print("  STANDALONE AUC  (single signal only, weight=1.0)")
    print("=" * 82)

    header = f"{'Signal':<22}"
    for tier in ALL_TIERS:
        header += f" {tier[:14]:>14}"
    print(header)
    print("-" * 82)

    for sig in signal_names:
        row = f"{sig:<22}"
        for tier in ALL_TIERS:
            solo = tier_results[tier]["standalone"][sig]
            row += f" {solo['auc']:>6.4f}({solo['ci_lower']:.2f}-{solo['ci_upper']:.2f})"
        print(row)
    print()


def print_delta_contribution_table(results: Dict[str, Any]) -> None:
    """Print delta contribution table with ranking."""
    signal_names = results["signal_names"]
    tier_results = results["tier_results"]
    weights = results["weights"]

    print("=" * 82)
    print("  DELTA CONTRIBUTION  (full_AUC - leave_one_out_AUC, positive = signal helps)")
    print("=" * 82)

    header = f"{'Signal':<22} {'Weight':>6}"
    for tier in ALL_TIERS:
        header += f" {tier[:14]:>14}"
    print(header)
    print("-" * 82)

    # Compute average delta across tiers for ranking
    avg_deltas = {}
    for sig in signal_names:
        deltas = [tier_results[t]["leave_one_out"][sig]["delta"] for t in ALL_TIERS]
        avg_deltas[sig] = sum(deltas) / len(deltas)

    ranked = sorted(signal_names, key=lambda s: avg_deltas[s], reverse=True)

    for sig in ranked:
        row = f"{sig:<22} {weights[sig]:>6.2f}"
        for tier in ALL_TIERS:
            delta = tier_results[tier]["leave_one_out"][sig]["delta"]
            row += f" {delta:>+14.4f}"
        row += f"  avg={avg_deltas[sig]:>+.4f}"
        print(row)
    print()


def print_correlation_matrix(results: Dict[str, Any]) -> None:
    """Print signal correlation matrix."""
    signal_names = results["signal_names"]
    corr = results["correlation_matrix"]

    print("=" * 82)
    print("  SIGNAL CORRELATION MATRIX  (Pearson r across all traces)")
    print("=" * 82)

    # Abbreviated names for compact display
    abbrev = {
        "causal_dag": "cDAG",
        "causal_concentration": "cConc",
        "cross_channel_mi": "xMI",
        "decay_type": "decay",
        "free_energy": "FE",
        "adaptation": "adapt",
    }

    header = f"{'':>22}"
    for sig in signal_names:
        header += f" {abbrev.get(sig, sig[:6]):>7}"
    print(header)
    print("-" * 82)

    for a in signal_names:
        row = f"{a:<22}"
        for b in signal_names:
            r = corr[a][b]
            row += f" {r:>7.3f}"
        print(row)
    print()


def print_full_composite_summary(results: Dict[str, Any]) -> None:
    """Print full composite AUC with CIs per tier."""
    tier_results = results["tier_results"]

    print("=" * 60)
    print("  FULL COMPOSITE AUC  (all 6 signals, original weights)")
    print("=" * 60)
    print(f"{'Tier':<22} {'AUC':>8} {'95% CI':>18}")
    print("-" * 60)

    for tier in ALL_TIERS:
        full = tier_results[tier]["full_composite"]
        ci = f"[{full['ci_lower']:.4f}, {full['ci_upper']:.4f}]"
        print(f"{tier:<22} {full['auc']:>8.4f} {ci:>18}")
    print()


def generate_latex_tables(results: Dict[str, Any]) -> str:
    """Generate LaTeX tables for paper inclusion."""
    signal_names = results["signal_names"]
    tier_results = results["tier_results"]
    weights = results["weights"]
    corr = results["correlation_matrix"]

    lines = []

    # Table 1: Leave-one-out AUC
    lines.append("% Leave-one-out AUC (tier x signal)")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Leave-one-out ablation: composite AUC with each signal removed (weights renormalized). "
                  "$\\Delta$ = full $-$ ablated; positive values indicate the signal contributes to discrimination.}")
    lines.append("\\label{tab:ablation-loo}")
    cols = "l" + "r" * (len(ALL_TIERS) * 2)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Signal"
    for tier in ALL_TIERS:
        label = tier.replace("_", " ").title()
        header += f" & AUC & $\\Delta$"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for sig in signal_names:
        label = sig.replace("_", "\\_")
        row = f"{label}"
        for tier in ALL_TIERS:
            loo = tier_results[tier]["leave_one_out"][sig]
            row += f" & {loo['auc']:.3f} & {loo['delta']:+.3f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\midrule")
    row = "Full composite"
    for tier in ALL_TIERS:
        full = tier_results[tier]["full_composite"]
        row += f" & {full['auc']:.3f} & ---"
    row += " \\\\"
    lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: Standalone AUC
    lines.append("% Standalone AUC (single signal only)")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Standalone signal AUC (single signal, weight=1.0) with 95\\% bootstrap CI.}")
    lines.append("\\label{tab:ablation-standalone}")
    cols = "lr" + "r" * len(ALL_TIERS)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Signal & $w$"
    for tier in ALL_TIERS:
        label = tier.replace("_", " ").title()
        header += f" & {label}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for sig in signal_names:
        label = sig.replace("_", "\\_")
        row = f"{label} & {weights[sig]:.2f}"
        for tier in ALL_TIERS:
            solo = tier_results[tier]["standalone"][sig]
            row += f" & {solo['auc']:.3f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 3: Correlation matrix
    lines.append("% Signal correlation matrix")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Pairwise Pearson correlation between normalized signal values.}")
    lines.append("\\label{tab:signal-correlation}")
    cols = "l" + "r" * len(signal_names)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    abbrev = {
        "causal_dag": "cDAG",
        "causal_concentration": "cConc",
        "cross_channel_mi": "xMI",
        "decay_type": "decay",
        "free_energy": "FE",
        "adaptation": "adapt",
    }
    header = ""
    for sig in signal_names:
        header += f" & {abbrev.get(sig, sig[:5])}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for a in signal_names:
        row = abbrev.get(a, a[:5])
        for b in signal_names:
            r = corr[a][b]
            if a == b:
                row += " & ---"
            else:
                row += f" & {r:.2f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Signal ablation study for consciousness-correlate discrimination",
    )
    parser.add_argument("-n", "--n-samples", type=int, default=50,
                        help="Traces per class (default: 50)")
    parser.add_argument("--n-events", type=int, default=30,
                        help="Events per trace (default: 30)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="JSON output path")
    parser.add_argument("--output-latex", type=Path, default=None,
                        help="LaTeX table output path")
    args = parser.parse_args()

    print(f"Signal ablation study: n={args.n_samples}, events={args.n_events}, "
          f"bootstrap={args.n_bootstrap}")

    evaluator = AdversarialEvaluator(seed=args.seed)
    results = evaluator.run_signal_ablation(
        n_samples=args.n_samples,
        n_events=args.n_events,
        n_bootstrap=args.n_bootstrap,
    )

    # Print all tables
    print_full_composite_summary(results)
    print_leave_one_out_table(results)
    print_standalone_table(results)
    print_delta_contribution_table(results)
    print_correlation_matrix(results)

    # JSON output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"JSON results saved to {args.output}")

    # LaTeX output
    if args.output_latex:
        args.output_latex.parent.mkdir(parents=True, exist_ok=True)
        latex = generate_latex_tables(results)
        args.output_latex.write_text(latex)
        print(f"LaTeX tables saved to {args.output_latex}")


if __name__ == "__main__":
    main()
