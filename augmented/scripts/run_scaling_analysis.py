#!/usr/bin/env python3
"""Scaling analysis: how detection AUC varies with trace length.

Generates authentic and forged traces at varying lengths, computes
AUC with bootstrap 95% CIs for each (trace_length, adversary_tier)
pair, and reports the saturation curve as a table.

Outputs:
  - Console table showing AUC (CI) per length x tier
  - Optional JSON (--output)
  - Optional LaTeX table (--output-latex)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List

from scholawrite.adversarial import (
    ALL_TIERS,
    AdversarialEvaluator,
    _bootstrap_auc,
    _extract_features,
    _generate_authentic_trace,
)

TIER_LABELS = {
    "naive": "Naive",
    "statistical": "Statistical",
    "reverse_engineered": "Rev-Eng",
    "expert": "Expert",
}


def _combined_score(features: Dict[str, float]) -> float:
    """Weighted combination matching AdversarialEvaluator.evaluate_forgery_detection."""
    return (
        0.3 * features["glucose_monotonicity"]
        + 0.25 * features["granger_asymmetry"]
        + 0.2 * abs(features["coupling"])
        + 0.15 * abs(features["latency_glucose_corr"])
        + 0.1 * features["locality"]
    )


def run_scaling_analysis(
    lengths: List[int],
    n_samples: int,
    seed: int,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Run scaling analysis across trace lengths.

    Returns dict keyed by trace length, each containing per-tier AUC + CIs.
    """
    evaluator = AdversarialEvaluator(seed=seed)
    results: Dict[str, Any] = {
        "n_samples": n_samples,
        "seed": seed,
        "lengths": lengths,
        "scaling": {},
    }

    for n_events in lengths:
        t0 = time.time()
        print(f"  n_events={n_events:>4d} ...", end="", flush=True)

        # Generate authentic traces
        authentic_traces = [
            _generate_authentic_trace(seed=seed + i, n_events=n_events)
            for i in range(n_samples)
        ]

        # Generate forged traces for all tiers
        traces_by_tier = evaluator._generate_tier_traces(
            n_samples, n_events, authentic_traces,
        )

        tier_results: Dict[str, Any] = {}
        for tier in ALL_TIERS:
            forged = traces_by_tier[tier]

            auth_features = [_extract_features(t) for t in authentic_traces]
            forge_features = [_extract_features(t) for t in forged]

            y_true = [1.0] * len(auth_features) + [0.0] * len(forge_features)
            y_score = [_combined_score(f) for f in auth_features + forge_features]

            auc_val, ci_lo, ci_hi = _bootstrap_auc(
                y_true, y_score,
                n_bootstrap=n_bootstrap, ci_level=ci_level,
            )
            tier_results[tier] = {
                "auc": auc_val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            }

        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s)")
        results["scaling"][n_events] = tier_results

    return results


def print_table(results: Dict[str, Any]) -> None:
    """Print scaling results as a formatted console table."""
    scaling = results["scaling"]
    lengths = results["lengths"]

    print(f"\n{'='*80}")
    print("  Scaling Analysis: AUC (95% CI) by Trace Length x Adversary Tier")
    print(f"  n_samples={results['n_samples']}, seed={results['seed']}")
    print(f"{'='*80}\n")

    # Header
    header = f"  {'Length':>6}"
    for tier in ALL_TIERS:
        header += f"  {TIER_LABELS.get(tier, tier):>22}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*22}" * len(ALL_TIERS))

    # Rows
    for n_events in lengths:
        tier_data = scaling.get(n_events, scaling.get(str(n_events), {}))
        row = f"  {n_events:>6}"
        for tier in ALL_TIERS:
            td = tier_data.get(tier, {})
            a = td.get("auc", 0.0)
            lo = td.get("ci_lower", 0.0)
            hi = td.get("ci_upper", 0.0)
            cell = f"{a:.3f} [{lo:.3f},{hi:.3f}]"
            row += f"  {cell:>22}"
        print(row)

    print()


def write_latex(results: Dict[str, Any], path: str) -> None:
    """Write scaling results as a LaTeX table."""
    scaling = results["scaling"]
    lengths = results["lengths"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection AUC (95\% CI) as a function of trace length.}",
        r"\label{tab:scaling}",
        r"\begin{tabular}{r" + "c" * len(ALL_TIERS) + "}",
        r"\toprule",
    ]

    # Header
    tier_cols = " & ".join(f"\\textbf{{{TIER_LABELS.get(t, t)}}}" for t in ALL_TIERS)
    lines.append(f"\\textbf{{Length}} & {tier_cols} \\\\")
    lines.append(r"\midrule")

    # Rows
    for n_events in lengths:
        tier_data = scaling.get(n_events, scaling.get(str(n_events), {}))
        cells = []
        for tier in ALL_TIERS:
            td = tier_data.get(tier, {})
            a = td.get("auc", 0.0)
            lo = td.get("ci_lower", 0.0)
            hi = td.get("ci_upper", 0.0)
            cell = f"{a:.3f}$_{{[{lo:.2f},{hi:.2f}]}}$"
            cells.append(cell)
        lines.append(f"{n_events} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX table written to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaling analysis: AUC vs. trace length across adversary tiers",
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=50,
        help="Number of traces per class per length (default: 50)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-l", "--lengths", type=str, default="10,20,30,50,100,200,300",
        help="Comma-separated trace lengths (default: 10,20,30,50,100,200,300)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap iterations for CI estimation (default: 1000)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-latex", type=str, default=None,
        help="Output LaTeX table file path",
    )
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",")]

    print("\n=== Scaling Analysis ===\n")
    print(f"  Trace lengths: {lengths}")
    print(f"  Samples/class: {args.n_samples}")
    print(f"  Seed:          {args.seed}")
    print(f"  Bootstrap:     {args.n_bootstrap}")
    print()

    results = run_scaling_analysis(
        lengths=lengths,
        n_samples=args.n_samples,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )

    print_table(results)

    if args.output:
        # JSON keys must be strings
        serializable = dict(results)
        serializable["scaling"] = {
            str(k): v for k, v in results["scaling"].items()
        }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  JSON written to: {args.output}")

    if args.output_latex:
        write_latex(results, args.output_latex)


if __name__ == "__main__":
    main()
