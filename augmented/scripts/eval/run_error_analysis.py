#!/usr/bin/env python
"""Per-injection-level error analysis: false negative breakdown.

Disaggregates detection errors by injection_level × adversary_tier
to identify where detection fails and which signal combinations
are responsible.

Usage:
    uv run python scripts/run_error_analysis.py -n 50
    uv run python scripts/run_error_analysis.py -n 100 --output-latex results/error_analysis.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scholawrite.adversarial import (
    AdversarialEvaluator,
    ForgedTraceGenerator,
    ALL_TIERS,
    _extract_consciousness_signals,
    _generate_authentic_trace,
    _ACADEMIC_WORDS,
)
from scholawrite.consciousness_signatures import _WEIGHTS, _HUMAN_THRESHOLD
from scholawrite.metrics import auc, bootstrap_auc_ci


def _classify(score: float, threshold: float = _HUMAN_THRESHOLD) -> str:
    """Classify as human-like or AI-like."""
    return "human" if score >= threshold else "ai"


def run_error_analysis(
    n_samples: int = 50,
    n_events: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute per-tier error breakdown with signal-level diagnosis."""
    results: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_events": n_events,
        "threshold": _HUMAN_THRESHOLD,
    }

    # Authentic traces
    authentic = [
        _generate_authentic_trace(seed=seed + i, n_events=n_events)
        for i in range(n_samples)
    ]
    auth_sigs = [_extract_consciousness_signals(t) for t in authentic]
    auth_composites = [sum(s.values()) / len(s) for s in auth_sigs]

    # True positive rate (authentic classified as human)
    tp = sum(1 for c in auth_composites if _classify(c) == "human")
    results["authentic"] = {
        "true_positive_rate": round(tp / n_samples, 4),
        "mean_composite": round(sum(auth_composites) / len(auth_composites), 4),
    }

    # Per-tier analysis
    tier_seeds = {"naive": 1000, "statistical": 2000, "reverse_engineered": 3000, "expert": 4000}
    sample_text = " ".join(_ACADEMIC_WORDS * 5)

    for tier in ALL_TIERS:
        offset = tier_seeds.get(tier, 5000)
        forged = []
        for i in range(n_samples):
            gen = ForgedTraceGenerator(seed=seed + offset + i)
            if tier == "naive":
                forged.append(gen.generate_naive_forgery(sample_text, n_events=n_events))
            elif tier == "statistical":
                template = authentic[i % len(authentic)]
                forged.append(gen.generate_statistical_forgery(sample_text, template))
            elif tier == "reverse_engineered":
                trace, _ = gen.generate_reverse_engineered_forgery(
                    sample_text, author_id=f"forger_{i}", n_events=n_events,
                )
                forged.append(trace)
            else:
                trace, _ = gen.generate_expert_forgery(
                    sample_text, author_id=f"expert_{i}", n_events=n_events,
                )
                forged.append(trace)

        forge_sigs = [_extract_consciousness_signals(t) for t in forged]
        forge_composites = [sum(s.values()) / len(s) for s in forge_sigs]

        # False negative: forged classified as human
        fn = sum(1 for c in forge_composites if _classify(c) == "human")
        fn_rate = fn / n_samples

        # AUC
        y_true = [1.0] * n_samples + [0.0] * n_samples
        scores = auth_composites + forge_composites
        pt, lo, hi = bootstrap_auc_ci(y_true, scores)

        # Per-signal analysis for false negatives
        fn_indices = [i for i, c in enumerate(forge_composites) if _classify(c) == "human"]
        signal_means_fn: Dict[str, float] = {}
        signal_means_all: Dict[str, float] = {}

        signal_names = list(auth_sigs[0].keys())
        for sig in signal_names:
            signal_means_all[sig] = round(
                sum(forge_sigs[i][sig] for i in range(n_samples)) / n_samples, 4
            )
            if fn_indices:
                signal_means_fn[sig] = round(
                    sum(forge_sigs[i][sig] for i in fn_indices) / len(fn_indices), 4
                )
            else:
                signal_means_fn[sig] = 0.0

        # Which signals contribute most to false negatives?
        signal_contribution: Dict[str, float] = {}
        auth_means = {
            sig: sum(auth_sigs[i][sig] for i in range(n_samples)) / n_samples
            for sig in signal_names
        }
        for sig in signal_names:
            if fn_indices:
                # How close are FN signals to authentic distribution?
                gap = abs(auth_means[sig] - signal_means_fn[sig])
                signal_contribution[sig] = round(gap, 4)
            else:
                signal_contribution[sig] = 0.0

        results[tier] = {
            "false_negative_rate": round(fn_rate, 4),
            "false_negative_count": fn,
            "auc": {"point": pt, "ci": [lo, hi]},
            "mean_composite_forged": round(sum(forge_composites) / len(forge_composites), 4),
            "signal_means_all_forged": signal_means_all,
            "signal_means_false_negatives": signal_means_fn,
            "signal_gap_from_authentic": signal_contribution,
        }

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted error analysis."""
    print(f"\n{'=' * 90}")
    print(f"  Error Analysis (n={results['n_samples']}, threshold={results['threshold']})")
    print(f"{'=' * 90}")

    print(f"\nAuthentic: TPR={results['authentic']['true_positive_rate']:.3f}, "
          f"mean composite={results['authentic']['mean_composite']:.3f}")

    print(f"\n{'Tier':<20} {'FN Rate':>8} {'FN Count':>10} {'AUC (95% CI)':>25} {'Mean Comp':>10}")
    print("-" * 80)

    for tier in ALL_TIERS:
        d = results[tier]
        a = d["auc"]
        print(
            f"{tier:<20} {d['false_negative_rate']:>8.3f} {d['false_negative_count']:>10d} "
            f"{a['point']:.3f} ({a['ci'][0]:.3f}-{a['ci'][1]:.3f}):>25 "
            f"{d['mean_composite_forged']:>10.3f}"
        )

    # Signal breakdown for expert tier
    if "expert" in results:
        d = results["expert"]
        print(f"\n--- Expert Tier Signal Breakdown ---")
        print(f"{'Signal':<25} {'All Forged':>12} {'False Neg':>12} {'Gap from Auth':>14}")
        print("-" * 65)
        for sig in d["signal_means_all_forged"]:
            print(
                f"{sig:<25} {d['signal_means_all_forged'][sig]:>12.4f} "
                f"{d['signal_means_false_negatives'][sig]:>12.4f} "
                f"{d['signal_gap_from_authentic'][sig]:>14.4f}"
            )
    print()


def _write_latex_table(results: Dict[str, Any], path: str) -> None:
    """Write booktabs LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection error analysis by adversary tier. FN Rate = "
        r"fraction of forged traces misclassified as human-like.}",
        r"\label{tab:error-analysis}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Tier & FN Rate & AUC (95\% CI) & Mean Composite \\",
        r"\midrule",
    ]

    for tier in ALL_TIERS:
        d = results[tier]
        a = d["auc"]
        label = tier.replace("_", r"\_")
        lines.append(
            f"{label} & {d['false_negative_rate']:.3f} & "
            f"{a['point']:.3f} ({a['ci'][0]:.3f}--{a['ci'][1]:.3f}) & "
            f"{d['mean_composite_forged']:.3f}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Per-injection-level error analysis")
    parser.add_argument("-n", "--n-samples", type=int, default=50)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("error_analysis.json"))
    parser.add_argument("--output-latex", type=str, default=None)
    args = parser.parse_args()

    results = run_error_analysis(args.n_samples, args.n_events, args.seed)
    print_results(results)

    if args.output_latex:
        _write_latex_table(results, args.output_latex)
        print(f"LaTeX table: {args.output_latex}")

    args.output.write_text(json.dumps(results, indent=2, default=float))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
