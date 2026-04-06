#!/usr/bin/env python
"""Signal ablation study for consciousness-correlate discrimination.

Runs ablation across all 4 adversary tiers and reports per-signal contribution
with bootstrap 95% confidence intervals.

Usage:
    uv run python scripts/run_ablation.py -n 50
    uv run python scripts/run_ablation.py -n 50 --output-latex results/ablation.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from scholawrite.adversarial import (
    AdversarialEvaluator,
    ForgedTraceGenerator,
    ALL_TIERS,
    _extract_consciousness_signals,
    _generate_authentic_trace,
    _ACADEMIC_WORDS,
)
from scholawrite.metrics import auc, bootstrap_auc_ci


def _fmt_auc_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} ({lo:.3f}-{hi:.3f})"


def run_ablation_for_tier(
    evaluator: AdversarialEvaluator,
    tier: str,
    n_samples: int,
    n_events: int,
) -> Dict[str, Any]:
    """Run signal ablation against a specific adversary tier, with CIs."""
    seed = evaluator.seed

    # Generate authentic traces
    authentic_traces = [
        _generate_authentic_trace(seed=seed + i, n_events=n_events)
        for i in range(n_samples)
    ]

    # Generate forged traces for the tier
    sample_text = " ".join(_ACADEMIC_WORDS * 3)
    forged_traces = []
    tier_seed_offset = {
        "naive": 1000, "statistical": 2000,
        "reverse_engineered": 3000, "expert": 4000,
    }
    offset = tier_seed_offset.get(tier, 5000)

    for i in range(n_samples):
        gen = ForgedTraceGenerator(seed=seed + offset + i)
        if tier == "naive":
            forged_traces.append(gen.generate_naive_forgery(sample_text, n_events=n_events))
        elif tier == "statistical":
            template = authentic_traces[i % len(authentic_traces)]
            forged_traces.append(gen.generate_statistical_forgery(sample_text, template))
        elif tier == "reverse_engineered":
            trace, _ = gen.generate_reverse_engineered_forgery(
                sample_text, author_id=f"forger_{i}", n_events=n_events,
            )
            forged_traces.append(trace)
        else:  # expert
            trace, _ = gen.generate_expert_forgery(
                sample_text, author_id=f"expert_{i}", n_events=n_events,
            )
            forged_traces.append(trace)

    # Extract signals
    auth_signals = [_extract_consciousness_signals(t) for t in authentic_traces]
    forge_signals = [_extract_consciousness_signals(t) for t in forged_traces]
    all_signals = auth_signals + forge_signals
    y_true = [1.0] * len(auth_signals) + [0.0] * len(forge_signals)

    signal_names = list(auth_signals[0].keys())

    # Baseline composite AUC with CI
    baseline_scores = [sum(s.values()) / len(s) for s in all_signals]
    baseline_auc_val = auc(y_true, baseline_scores)
    baseline_pt, baseline_lo, baseline_hi = bootstrap_auc_ci(y_true, baseline_scores)

    # Ablation
    ablation: Dict[str, Dict[str, Any]] = {}
    deltas: List[Tuple[str, float]] = []

    for sig in signal_names:
        remaining = [n for n in signal_names if n != sig]
        scores_without = [sum(s[r] for r in remaining) / len(remaining) for s in all_signals]
        auc_without = auc(y_true, scores_without)
        pt_wo, lo_wo, hi_wo = bootstrap_auc_ci(y_true, scores_without)

        scores_only = [s[sig] for s in all_signals]
        auc_only = auc(y_true, scores_only)
        pt_only, lo_only, hi_only = bootstrap_auc_ci(y_true, scores_only)

        delta = baseline_auc_val - auc_without
        ablation[sig] = {
            "auc_without": round(auc_without, 4),
            "auc_without_ci": (pt_wo, lo_wo, hi_wo),
            "auc_only": round(auc_only, 4),
            "auc_only_ci": (pt_only, lo_only, hi_only),
            "delta_without": round(delta, 4),
            "contribution_rank": 0,
        }
        deltas.append((sig, delta))

    deltas.sort(key=lambda x: x[1], reverse=True)
    for rank, (sig, _) in enumerate(deltas, start=1):
        ablation[sig]["contribution_rank"] = rank

    return {
        "baseline_auc": round(baseline_auc_val, 4),
        "baseline_ci": (baseline_pt, baseline_lo, baseline_hi),
        "ablation": ablation,
        "n_samples": n_samples,
        "n_events": n_events,
    }


def print_table(tier: str, result: Dict[str, Any]) -> None:
    """Print a formatted ablation table for one tier with CIs."""
    bl_pt, bl_lo, bl_hi = result["baseline_ci"]
    print(f"\n{'=' * 100}")
    print(f"  Tier: {tier}    (baseline AUC = {_fmt_auc_ci(bl_pt, bl_lo, bl_hi)})")
    print(f"{'=' * 100}")
    header = (
        f"{'Signal':<25} {'AUC w/o (95% CI)':>25} "
        f"{'AUC only (95% CI)':>25} {'Delta':>9} {'Rank':>5}"
    )
    print(header)
    print("-" * 100)

    # Sort by contribution rank
    sorted_sigs = sorted(
        result["ablation"].items(),
        key=lambda x: x[1]["contribution_rank"],
    )
    for sig, data in sorted_sigs:
        wo_pt, wo_lo, wo_hi = data["auc_without_ci"]
        only_pt, only_lo, only_hi = data["auc_only_ci"]
        print(
            f"{sig:<25} {_fmt_auc_ci(wo_pt, wo_lo, wo_hi):>25} "
            f"{_fmt_auc_ci(only_pt, only_lo, only_hi):>25} "
            f"{data['delta_without']:>+9.4f} {data['contribution_rank']:>5d}"
        )
    print()


TIER_LATEX_LABELS = {
    "naive": "Naive",
    "statistical": "Statistical",
    "reverse_engineered": "Rev-Eng",
    "expert": "Expert",
}


def _write_latex_table(all_results: Dict[str, Any], path: str) -> None:
    """Write ablation results as a booktabs LaTeX table.

    One table per tier showing AUC-only with CI for each signal.
    """
    tiers = ALL_TIERS
    # Get signal names from first tier
    first_tier = next(iter(all_results.values()))
    signals = sorted(
        first_tier["ablation"].keys(),
        key=lambda s: first_tier["ablation"][s]["contribution_rank"],
    )

    n_tiers = len(tiers)
    col_spec = "l" + "c" * n_tiers
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Signal ablation: AUC (95\% CI) using only the given signal, by adversary tier.}",
        r"\label{tab:signal-ablation}",
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
            data = all_results[t]["ablation"][sig]
            pt, lo, hi = data["auc_only_ci"]
            cell = rf"{pt:.3f} ({lo:.3f}--{hi:.3f})"
            if pt < 0.55:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    # Baseline row
    lines.append(r"\midrule")
    cells = ["Baseline (all)"]
    for t in tiers:
        pt, lo, hi = all_results[t]["baseline_ci"]
        cell = rf"{pt:.3f} ({lo:.3f}--{hi:.3f})"
        cells.append(cell)
    lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Signal ablation study")
    parser.add_argument("-n", "--n-samples", type=int, default=100, help="Samples per class")
    parser.add_argument("--n-events", type=int, default=30, help="Events per trace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("signal_ablation_results.json"))
    parser.add_argument("--output-latex", type=str, default=None, help="Path for booktabs LaTeX table")
    args = parser.parse_args()

    evaluator = AdversarialEvaluator(seed=args.seed)
    all_results: Dict[str, Any] = {}

    for tier in ALL_TIERS:
        print(f"Running ablation for tier: {tier} ...")
        result = run_ablation_for_tier(evaluator, tier, args.n_samples, args.n_events)
        all_results[tier] = result
        print_table(tier, result)

    # Serialize: convert CI tuples to lists for JSON
    json_results: Dict[str, Any] = {}
    for tier, result in all_results.items():
        tier_copy = dict(result)
        tier_copy["baseline_ci"] = list(result["baseline_ci"])
        tier_copy["ablation"] = {}
        for sig, data in result["ablation"].items():
            data_copy = dict(data)
            data_copy["auc_without_ci"] = list(data["auc_without_ci"])
            data_copy["auc_only_ci"] = list(data["auc_only_ci"])
            tier_copy["ablation"][sig] = data_copy
        json_results[tier] = tier_copy

    # Write LaTeX table
    if args.output_latex:
        _write_latex_table(all_results, args.output_latex)
        print(f"LaTeX table written to: {args.output_latex}")

    # Save JSON
    args.output.write_text(json.dumps(json_results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
