#!/usr/bin/env python3
"""CLI script for the Impossible Forgery Experiment.

Runs the full adversarial evaluation and outputs:
1. Legacy per-strategy AUC (5 core features)
2. Signal x tier AUC matrix (6 consciousness-correlate signals) with 95% CIs

The signal x tier matrix is the primary output for the paper,
showing how each consciousness signal degrades across adversary tiers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from scholawrite.adversarial import (
    AdversarialEvaluator,
    ALL_TIERS,
    _extract_consciousness_signals,
    _extract_features,
    _generate_authentic_trace,
)
from scholawrite.consciousness_signatures import _WEIGHTS as CONSCIOUSNESS_WEIGHTS
from scholawrite.metrics import auc, bootstrap_auc_ci

TIER_LABELS = {
    "naive": "Naive",
    "statistical": "Statistical",
    "reverse_engineered": "Rev-Eng",
    "expert": "Expert",
}

TIER_LATEX_LABELS = TIER_LABELS


def _fmt_auc_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} ({lo:.3f}-{hi:.3f})"


def _compute_signal_tier_cis(
    evaluator: AdversarialEvaluator,
    n_samples: int,
    n_events: int,
) -> Dict[str, Any]:
    """Recompute signal x tier matrix with bootstrap CIs.

    Returns dict with signal_tier_matrix_ci, composite_by_tier_ci,
    and raw y_true/signal scores for each tier.
    """
    seed = evaluator.seed

    # Generate authentic traces
    authentic_traces = [
        _generate_authentic_trace(seed=seed + i, n_events=n_events)
        for i in range(n_samples)
    ]

    # Generate forged traces for all tiers
    traces_by_tier = evaluator._generate_tier_traces(n_samples, n_events, authentic_traces)

    signal_tier_ci: Dict[str, Dict[str, Dict[str, float]]] = {}
    composite_ci: Dict[str, Dict[str, float]] = {}

    for tier in ALL_TIERS:
        forged = traces_by_tier[tier]

        auth_signals = [_extract_consciousness_signals(t) for t in authentic_traces]
        forge_signals = [_extract_consciousness_signals(t) for t in forged]
        all_signals = auth_signals + forge_signals
        y_true = [1.0] * len(auth_signals) + [0.0] * len(forge_signals)

        signal_names = list(auth_signals[0].keys())

        for sig in signal_names:
            scores = [s[sig] for s in all_signals]
            pt, lo, hi = bootstrap_auc_ci(y_true, scores)
            signal_tier_ci.setdefault(sig, {})[tier] = {
                "auc": pt, "ci_lower": lo, "ci_upper": hi,
            }

        # Composite
        composite_scores = [
            sum(CONSCIOUSNESS_WEIGHTS.get(k, 0.0) * v for k, v in s.items())
            for s in all_signals
        ]
        pt, lo, hi = bootstrap_auc_ci(y_true, composite_scores)
        composite_ci[tier] = {"auc": pt, "ci_lower": lo, "ci_upper": hi}

    return {
        "signal_tier_matrix_ci": signal_tier_ci,
        "composite_by_tier_ci": composite_ci,
    }


def _compute_legacy_cis(
    evaluator: AdversarialEvaluator,
    n_samples: int,
    n_events: int,
) -> Dict[str, Dict[str, float]]:
    """Compute CIs for the legacy 5-feature combined AUC per strategy."""
    seed = evaluator.seed
    authentic_traces = [
        _generate_authentic_trace(seed=seed + i, n_events=n_events)
        for i in range(n_samples)
    ]
    traces_by_tier = evaluator._generate_tier_traces(n_samples, n_events, authentic_traces)

    cis: Dict[str, Dict[str, float]] = {}
    for tier, forged in traces_by_tier.items():
        auth_features = [_extract_features(t) for t in authentic_traces]
        forge_features = [_extract_features(t) for t in forged]

        y_true = [1.0] * len(auth_features) + [0.0] * len(forge_features)
        y_score: List[float] = []
        for features in auth_features + forge_features:
            score = (
                0.3 * features["glucose_monotonicity"]
                + 0.25 * features["granger_asymmetry"]
                + 0.2 * abs(features["coupling"])
                + 0.15 * abs(features["latency_glucose_corr"])
                + 0.1 * features["locality"]
            )
            y_score.append(score)

        pt, lo, hi = bootstrap_auc_ci(y_true, y_score)
        strategy_key = f"{tier}_forgery"
        cis[strategy_key] = {"auc": pt, "ci_lower": lo, "ci_upper": hi}

    return cis


def _print_signal_tier_matrix(consciousness: dict, ci_data: dict) -> None:
    """Print the signal x tier AUC matrix with CIs."""
    matrix = consciousness.get("signal_tier_matrix", {})
    composite = consciousness.get("composite_by_tier", {})
    signal_ci = ci_data.get("signal_tier_matrix_ci", {})
    composite_ci = ci_data.get("composite_by_tier_ci", {})

    if not matrix:
        print("  (no consciousness signal data)")
        return

    signals = list(matrix.keys())
    tiers = ALL_TIERS

    # Header
    header = f"  {'Signal':<25}"
    for t in tiers:
        header += f" {TIER_LABELS.get(t, t):>22}"
    print(header)
    print(f"  {'-'*25}" + f" {'-'*22}" * len(tiers))

    # Signal rows
    for sig in signals:
        row = f"  {sig:<25}"
        for t in tiers:
            ci = signal_ci.get(sig, {}).get(t, {})
            pt = ci.get("auc", matrix[sig].get(t, 0.0))
            lo = ci.get("ci_lower", pt)
            hi = ci.get("ci_upper", pt)
            marker = "*" if pt < 0.55 else " "
            row += f" {_fmt_auc_ci(pt, lo, hi):>21}{marker}"
        print(row)

    # Composite row
    print(f"  {'-'*25}" + f" {'-'*22}" * len(tiers))
    row = f"  {'COMPOSITE':<25}"
    for t in tiers:
        ci = composite_ci.get(t, {})
        pt = ci.get("auc", composite.get(t, 0.0))
        lo = ci.get("ci_lower", pt)
        hi = ci.get("ci_upper", pt)
        marker = "*" if pt < 0.55 else " "
        row += f" {_fmt_auc_ci(pt, lo, hi):>21}{marker}"
    print(row)
    print()
    print("  * = AUC < 0.55 (near coin-flip; signal defeated)")


def _write_latex_table(
    consciousness: dict,
    ci_data: dict,
    path: str,
) -> None:
    """Write signal x tier matrix as a booktabs LaTeX table."""
    matrix = consciousness.get("signal_tier_matrix", {})
    composite = consciousness.get("composite_by_tier", {})
    signal_ci = ci_data.get("signal_tier_matrix_ci", {})
    composite_ci = ci_data.get("composite_by_tier_ci", {})
    tiers = ALL_TIERS
    signals = list(matrix.keys())

    n_tiers = len(tiers)
    col_spec = "l" + "c" * n_tiers
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Signal $\times$ adversary tier AUC (95\% CI). Bold = defeated ($<0.55$).}",
        r"\label{tab:signal-tier-matrix}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]
    header_cells = ["Signal"] + [TIER_LATEX_LABELS.get(t, t) for t in tiers]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for sig in signals:
        sig_label = sig.replace("_", r"\_")
        cells = [sig_label]
        for t in tiers:
            ci = signal_ci.get(sig, {}).get(t, {})
            pt = ci.get("auc", matrix[sig].get(t, 0.0))
            lo = ci.get("ci_lower", pt)
            hi = ci.get("ci_upper", pt)
            cell = rf"{pt:.3f} ({lo:.3f}--{hi:.3f})"
            if pt < 0.55:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    cells = ["Composite"]
    for t in tiers:
        ci = composite_ci.get(t, {})
        pt = ci.get("auc", composite.get(t, 0.0))
        lo = ci.get("ci_lower", pt)
        hi = ci.get("ci_upper", pt)
        cell = rf"{pt:.3f} ({lo:.3f}--{hi:.3f})"
        if pt < 0.55:
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Impossible Forgery Experiment"
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=100,
        help="Number of traces per class (default: 100)",
    )
    parser.add_argument(
        "-e", "--n-events", type=int, default=30,
        help="Number of events per trace (default: 30)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-consciousness", action="store_true",
        help="Skip consciousness signal evaluation (faster)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--output-latex", type=str, default=None,
        help="Path for booktabs LaTeX table",
    )
    args = parser.parse_args()

    evaluator = AdversarialEvaluator(seed=args.seed)
    results = evaluator.run_full_evaluation(
        n_samples=args.n_samples,
        n_events=args.n_events,
        include_consciousness=not args.no_consciousness,
    )

    # Print legacy summary
    print("\n=== Impossible Forgery Experiment Results ===\n")
    print(f"  Samples per class: {results['n_samples']}")
    print(f"  Events per trace:  {results['n_events']}")
    print()

    # Compute legacy CIs
    print("  Computing bootstrap 95% CIs for legacy features...")
    legacy_cis = _compute_legacy_cis(evaluator, args.n_samples, args.n_events)

    print("  --- Core Feature Detection (5 features) ---\n")
    print(f"  {'Strategy':<30} {'Combined AUC (95% CI)':>30}  {'Best Feature':>20} {'Feature AUC':>12}")
    print(f"  {'-'*30} {'-'*30}  {'-'*20} {'-'*12}")

    for strategy_key in ["naive_forgery", "statistical_forgery",
                         "reverse_engineered_forgery", "expert_forgery"]:
        r = results.get(strategy_key, {})
        if not r:
            continue
        combined = r.get("combined_auc", 0.0)
        per_feat = r.get("per_feature_auc", {})
        if per_feat:
            best_feat = max(per_feat, key=per_feat.get)
            best_auc = per_feat[best_feat]
        else:
            best_feat = "N/A"
            best_auc = 0.0

        ci = legacy_cis.get(strategy_key, {})
        lo = ci.get("ci_lower", combined)
        hi = ci.get("ci_upper", combined)

        label = strategy_key.replace("_", " ").title()
        print(f"  {label:<30} {_fmt_auc_ci(combined, lo, hi):>30}  {best_feat:>20} {best_auc:>12.4f}")

    # Print consciousness signal x tier matrix with CIs
    ci_data: Dict[str, Any] = {}
    if "consciousness" in results:
        print()
        print("  Computing bootstrap 95% CIs for consciousness signals...")
        ci_data = _compute_signal_tier_cis(evaluator, args.n_samples, args.n_events)

        print()
        print("  --- Consciousness Signal x Adversary Tier AUC (95% CI) ---\n")
        _print_signal_tier_matrix(results["consciousness"], ci_data)

    print()

    # Write LaTeX table
    if args.output_latex and "consciousness" in results:
        _write_latex_table(results["consciousness"], ci_data, args.output_latex)
        print(f"  LaTeX table written to: {args.output_latex}")

    # Augment results JSON with CIs
    if ci_data and "consciousness" in results:
        results["consciousness"]["signal_tier_matrix_ci"] = ci_data.get("signal_tier_matrix_ci", {})
        results["consciousness"]["composite_by_tier_ci"] = ci_data.get("composite_by_tier_ci", {})
    if legacy_cis:
        results["legacy_cis"] = legacy_cis

    # Write JSON
    report = json.dumps(results, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"  Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
