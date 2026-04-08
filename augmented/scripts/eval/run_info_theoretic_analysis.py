#!/usr/bin/env python
"""Information-theoretic lower bound analysis via Fano's inequality.

Computes:
1. Fano bound on minimum mutual information for reliable detection
2. Critical trace length where detection becomes reliable
3. Per-signal conditional mutual information proxies
4. Independence structure of consciousness signals

Usage:
    uv run python scripts/run_info_theoretic_analysis.py -n 50
    uv run python scripts/run_info_theoretic_analysis.py -n 100 --output-latex results/info_theory.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scholawrite.adversarial import (
    ALL_TIERS,
    ForgedTraceGenerator,
    _extract_consciousness_signals,
    _generate_authentic_trace,
    _ACADEMIC_WORDS,
)
from scholawrite.metrics import (
    auc,
    bootstrap_auc_ci,
    compute_fano_bound,
    compute_conditional_auc,
)


def run_critical_length_analysis(
    n_samples: int = 50,
    seed: int = 42,
    lengths: List[int] | None = None,
) -> Dict[str, Any]:
    """Find critical trace length where detection becomes reliable."""
    if lengths is None:
        lengths = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    sample_text = " ".join(_ACADEMIC_WORDS * 10)
    results: Dict[str, Any] = {"n_samples": n_samples, "lengths": lengths}

    for tier in ALL_TIERS:
        tier_results: Dict[str, Any] = {}
        tier_seeds = {"naive": 1000, "statistical": 2000, "reverse_engineered": 3000, "expert": 4000}
        offset = tier_seeds.get(tier, 5000)

        for n_events in lengths:
            # Generate traces at this length
            authentic = [
                _generate_authentic_trace(seed=seed + i, n_events=n_events)
                for i in range(n_samples)
            ]
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
                        sample_text, author_id=f"f_{i}", n_events=n_events,
                    )
                    forged.append(trace)
                else:
                    trace, _ = gen.generate_expert_forgery(
                        sample_text, author_id=f"e_{i}", n_events=n_events,
                    )
                    forged.append(trace)

            # Extract signals
            auth_sigs = [_extract_consciousness_signals(t) for t in authentic]
            forge_sigs = [_extract_consciousness_signals(t) for t in forged]
            y_true = [1.0] * n_samples + [0.0] * n_samples

            # Build per-signal score lists
            signal_names = list(auth_sigs[0].keys())
            signal_scores: Dict[str, List[float]] = {
                sig: [auth_sigs[i][sig] for i in range(n_samples)]
                     + [forge_sigs[i][sig] for i in range(n_samples)]
                for sig in signal_names
            }

            # Conditional AUC analysis (includes Fano bound)
            cond = compute_conditional_auc(y_true, signal_scores)

            # Bootstrap CI on full AUC
            full_scores = [sum(s.values()) / len(s) for s in auth_sigs + forge_sigs]
            pt, lo, hi = bootstrap_auc_ci(y_true, full_scores)

            tier_results[str(n_events)] = {
                "auc": pt,
                "ci": [lo, hi],
                "error_rate": cond["error_rate"],
                "fano_mi_bound": cond["fano_mi_bound"],
                "per_signal": cond["signals"],
            }

        # Find critical length (AUC > 0.8 threshold)
        critical = None
        for n_events in lengths:
            if tier_results[str(n_events)]["auc"] > 0.8:
                critical = n_events
                break

        results[tier] = {
            "length_results": tier_results,
            "critical_length": critical,
        }

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted results."""
    print(f"\n{'=' * 100}")
    print(f"  Information-Theoretic Analysis (n={results['n_samples']})")
    print(f"{'=' * 100}")

    for tier in ALL_TIERS:
        d = results[tier]
        crit = d["critical_length"]
        crit_str = str(crit) if crit else ">200"
        print(f"\n--- {tier} (critical length: {crit_str}) ---")
        print(f"{'Length':>8} {'AUC (95% CI)':>25} {'Pe':>8} {'Fano MI (bits)':>15}")
        print("-" * 60)

        for n_events in results["lengths"]:
            r = d["length_results"][str(n_events)]
            print(
                f"{n_events:>8d} "
                f"{r['auc']:.3f} ({r['ci'][0]:.3f}-{r['ci'][1]:.3f}):>25 "
                f"{r['error_rate']:>8.3f} "
                f"{r['fano_mi_bound']:>15.4f}"
            )

    # Per-signal Fano analysis at longest length
    longest = str(results["lengths"][-1])
    for tier in ["naive", "expert"]:
        d = results[tier]["length_results"][longest]
        print(f"\n--- Per-Signal Analysis ({tier}, n_events={longest}) ---")
        print(f"{'Signal':<25} {'AUC':>8} {'Marginal':>10} {'Fano MI':>10}")
        print("-" * 55)
        for sig, info in d["per_signal"].items():
            print(
                f"{sig:<25} {info['standalone_auc']:>8.3f} "
                f"{info['marginal_contribution']:>+10.4f} "
                f"{info['fano_mi_needed']:>10.4f}"
            )
    print()


def _write_latex_table(results: Dict[str, Any], path: str) -> None:
    """Write booktabs LaTeX table showing critical lengths."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Critical trace length $n^*$ and Fano lower bound on mutual information "
        r"by adversary tier. $n^*$ = minimum events for AUC $> 0.8$.}",
        r"\label{tab:info-theoretic}",
        r"\begin{tabular}{l" + "c" * len(results["lengths"]) + "}",
        r"\toprule",
    ]

    header = "Tier"
    for n in results["lengths"]:
        header += f" & {n}"
    header += r" & $n^*$ \\"
    lines.append(header)
    lines.append(r"\midrule")

    for tier in ALL_TIERS:
        d = results[tier]
        label = tier.replace("_", r"\_")
        row = label
        for n in results["lengths"]:
            r = d["length_results"][str(n)]
            val = f"{r['auc']:.2f}"
            if r["auc"] > 0.8:
                val = rf"\textbf{{{val}}}"
            row += f" & {val}"
        crit = d["critical_length"]
        crit_str = str(crit) if crit else r"$>$200"
        row += f" & {crit_str}" + r" \\"
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Information-theoretic lower bound analysis")
    parser.add_argument("-n", "--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("info_theoretic_analysis.json"))
    parser.add_argument("--output-latex", type=str, default=None)
    args = parser.parse_args()

    results = run_critical_length_analysis(args.n_samples, args.seed)
    print_results(results)

    if args.output_latex:
        _write_latex_table(results, args.output_latex)
        print(f"LaTeX table: {args.output_latex}")

    args.output.write_text(json.dumps(results, indent=2, default=float))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
