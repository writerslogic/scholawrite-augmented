#!/usr/bin/env python3
"""Cross-detector comparison for the adversary hierarchy.

Evaluates internal detectors (NCD, CausalCoupling, ConsciousnessSignatures)
and external detectors (GPTZero, Originality.ai) against all adversary tiers.

Produces the adversary hierarchy table for the paper:
    Detector x Adversary Tier -> AUC (95% CI)

Usage:
    # Internal detectors only (no API keys needed)
    uv run python scripts/run_detector_comparison.py -n 20

    # With external APIs
    GPTZERO_API_KEY=... ORIGINALITY_API_KEY=... \
        uv run python scripts/run_detector_comparison.py -n 20 --external

    # Save results + LaTeX table
    uv run python scripts/run_detector_comparison.py -n 50 -o results/hierarchy.json --output-latex results/hierarchy.tex
"""
from __future__ import annotations

import argparse
import json
import os
import random

from scholawrite.adversarial import (
    AdversarialEvaluator,
    ForgedTraceGenerator,
    _generate_authentic_trace,
    ALL_TIERS,
    TIER_NAIVE,
    TIER_STATISTICAL,
    TIER_REVERSE_ENGINEERED,
    TIER_EXPERT,
)
from scholawrite.detector_harness import (
    CausalCouplingDetector,
    ConsciousnessDetector,
    DetectorHarness,
    GptZeroDetector,
    NcdDetector,
    OriginalityDetector,
    ADVERSARY_HIERARCHY,
)
from scholawrite.metrics import bootstrap_auc_ci

_ACADEMIC_WORDS = [
    "The", "methodology", "establishes", "a", "robust", "baseline",
    "for", "comparative", "analysis", "of", "emerging", "patterns",
    "in", "empirical", "research", "paradigm", "framework", "this",
    "study", "demonstrates", "that", "underlying", "assumptions",
]

TIER_LABELS = {
    TIER_NAIVE: "Naive(L0)",
    TIER_STATISTICAL: "Stat(L1)",
    TIER_REVERSE_ENGINEERED: "RevEng(L1)",
    TIER_EXPERT: "Expert(L2)",
}

TIER_LATEX_LABELS = {
    TIER_NAIVE: "Naive",
    TIER_STATISTICAL: "Statistical",
    TIER_REVERSE_ENGINEERED: "Rev-Eng",
    TIER_EXPERT: "Expert",
}


def _trace_to_text(trace) -> str:
    """Convert a trace to text by joining actual outputs."""
    return " ".join(e.actual_output for e in trace if e.actual_output)


def _generate_human_like_text(seed: int, n_words: int = 80) -> str:
    """Generate plausible academic text from authentic trace."""
    trace = _generate_authentic_trace(seed=seed, n_events=n_words)
    return _trace_to_text(trace)


def _fmt_auc_ci(point: float, lo: float, hi: float) -> str:
    """Format AUC with CI as 'AUC (lo-hi)'."""
    return f"{point:.3f} ({lo:.3f}-{hi:.3f})"


def _compute_ci_matrix(results: dict) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Compute bootstrap CIs for every detector x tier cell."""
    detail = results["detail"]
    ci_matrix: dict[str, dict[str, tuple[float, float, float]]] = {}
    for det_name, tier_data in detail.items():
        ci_matrix[det_name] = {}
        for tier, data in tier_data.items():
            y_true = data.get("y_true", [])
            y_score = data.get("y_score", [])
            if y_true and y_score:
                ci_matrix[det_name][tier] = bootstrap_auc_ci(y_true, y_score)
            else:
                val = data.get("auc", 0.5)
                ci_matrix[det_name][tier] = (val, val, val)
    return ci_matrix


def _write_latex_table(
    ci_matrix: dict,
    tiers: list[str],
    detectors: dict,
    path: str,
) -> None:
    """Write a booktabs-format LaTeX table."""
    n_tiers = len(tiers)
    col_spec = "l" + "c" * n_tiers
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-detector adversary hierarchy: AUC (95\% CI).}",
        r"\label{tab:detector-hierarchy}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]
    # Header row
    header_cells = ["Detector"] + [TIER_LATEX_LABELS.get(t, t) for t in tiers]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for det_name, tier_cis in ci_matrix.items():
        cells = [det_name.replace("_", r"\_")]
        for t in tiers:
            if t in tier_cis:
                pt, lo, hi = tier_cis[t]
                cell = rf"{pt:.3f} ({lo:.3f}--{hi:.3f})"
                if pt < 0.55:
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-detector adversary hierarchy evaluation")
    parser.add_argument("-n", "--n-samples", type=int, default=100)
    parser.add_argument("-e", "--n-events", type=int, default=30)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--external", action="store_true", help="Include GPTZero/Originality.ai")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--output-latex", type=str, default=None, help="Path for booktabs LaTeX table")
    args = parser.parse_args()

    print("\n=== Cross-Detector Adversary Hierarchy Evaluation ===\n")

    # Generate traces
    print(f"  Generating {args.n_samples} authentic traces...")
    human_traces = [
        _generate_authentic_trace(seed=args.seed + i, n_events=args.n_events)
        for i in range(args.n_samples)
    ]

    print(f"  Generating forged traces for {len(ALL_TIERS)} adversary tiers...")
    evaluator = AdversarialEvaluator(seed=args.seed)
    machine_traces_by_tier = evaluator._generate_tier_traces(
        args.n_samples, args.n_events, human_traces,
    )

    # Convert to texts for external detectors
    human_texts = [_trace_to_text(t) for t in human_traces]
    machine_texts_by_tier = {
        tier: [_trace_to_text(t) for t in traces]
        for tier, traces in machine_traces_by_tier.items()
    }

    # Build detector set
    harness = DetectorHarness()
    harness.add_detector(NcdDetector(reference_text=human_texts[0] if human_texts else ""))
    harness.add_detector(CausalCouplingDetector())
    harness.add_detector(ConsciousnessDetector())

    if args.external:
        gptzero_key = os.environ.get("GPTZERO_API_KEY", "")
        originality_key = os.environ.get("ORIGINALITY_API_KEY", "")

        if gptzero_key:
            harness.add_detector(GptZeroDetector(api_key=gptzero_key))
            print("  GPTZero API: enabled")
        else:
            print("  GPTZero API: GPTZERO_API_KEY not set, skipping")

        if originality_key:
            harness.add_detector(OriginalityDetector(api_key=originality_key))
            print("  Originality.ai API: enabled")
        else:
            print("  Originality.ai API: ORIGINALITY_API_KEY not set, skipping")

    # Run evaluation
    print("\n  Evaluating detectors...")
    results = harness.evaluate_on_traces(
        human_traces=human_traces,
        machine_traces_by_tier=machine_traces_by_tier,
        human_texts=human_texts,
        machine_texts_by_tier=machine_texts_by_tier,
    )

    # Compute bootstrap CIs for each cell
    print("  Computing bootstrap 95% CIs...")
    ci_matrix = _compute_ci_matrix(results)

    # Print the adversary hierarchy table with CIs
    print("\n  --- Adversary Hierarchy Table: Detector x Tier AUC (95% CI) ---\n")

    tiers = results["tiers"]
    detectors = results["detectors"]

    # Header
    header = f"  {'Detector':<25} {'Type':<18} {'Survives':<10}"
    for t in tiers:
        header += f" {TIER_LABELS.get(t, t):>22}"
    print(header)
    print(f"  {'-'*25} {'-'*18} {'-'*10}" + f" {'-'*22}" * len(tiers))

    # Rows
    for det_name in ci_matrix:
        meta = detectors.get(det_name, {})
        sig_type = meta.get("signal_type", "?")
        row = f"  {det_name:<25} {sig_type:<18} L{meta.get('hierarchy_level_survived', '?'):<9}"
        for t in tiers:
            ci = ci_matrix[det_name].get(t)
            if ci is None:
                row += f" {'N/A':>22}"
            else:
                pt, lo, hi = ci
                marker = "*" if pt < 0.55 else " "
                row += f" {_fmt_auc_ci(pt, lo, hi):>21}{marker}"
        print(row)

    print()
    print("  * = AUC < 0.55 (near coin-flip; detector defeated at this tier)")
    print()

    # Print hierarchy level explanations
    print("  --- Adversary Hierarchy Levels ---\n")
    for level in ADVERSARY_HIERARCHY:
        print(f"  L{level.level}: {level.name:<20} -- {level.description}")
    print()

    # Write LaTeX table
    if args.output_latex:
        _write_latex_table(ci_matrix, tiers, detectors, args.output_latex)
        print(f"  LaTeX table written to: {args.output_latex}")

    # Write JSON (strip raw scores for serialization)
    output_detail = {}
    for det, tier_data in results["detail"].items():
        output_detail[det] = {}
        for tier, data in tier_data.items():
            entry = {k: v for k, v in data.items() if k not in ("y_true", "y_score")}
            ci = ci_matrix.get(det, {}).get(tier)
            if ci:
                entry["ci_lower"] = ci[1]
                entry["ci_upper"] = ci[2]
            output_detail[det][tier] = entry

    # Build CI-augmented matrix for JSON
    matrix_ci = {}
    for det_name, tier_cis in ci_matrix.items():
        matrix_ci[det_name] = {}
        for t, (pt, lo, hi) in tier_cis.items():
            matrix_ci[det_name][t] = {"auc": pt, "ci_lower": lo, "ci_upper": hi}

    output = {
        "config": {
            "n_samples": args.n_samples,
            "n_events": args.n_events,
            "seed": args.seed,
            "external_apis": args.external,
        },
        "matrix": results["matrix"],
        "matrix_ci": matrix_ci,
        "tiers": results["tiers"],
        "detectors": results["detectors"],
        "detail": output_detail,
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
