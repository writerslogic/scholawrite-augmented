#!/usr/bin/env python
"""Cross-domain evaluation: train on academic, test on other domains.

Tests whether consciousness-correlate signals generalize beyond academic
writing to fiction, journalism, and technical documentation.

Usage:
    uv run python scripts/run_domain_transfer.py -n 50
    uv run python scripts/run_domain_transfer.py -n 100 --output-latex results/domain_transfer.tex
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scholawrite.adversarial import (
    ForgedTraceGenerator,
    ALL_TIERS,
    _extract_consciousness_signals,
    _ACADEMIC_WORDS,
)
from scholawrite.causal_core import IrreversibleProcessEngine, LexicalIntention
from scholawrite.embodied import EmbodiedScholar
from scholawrite.schema import CausalEvent
from scholawrite.metrics import auc, bootstrap_auc_ci


# ── Domain-specific vocabularies and cognitive profiles ──────────────────

DOMAINS = {
    "academic": {
        "vocab": _ACADEMIC_WORDS,
        "depth_range": (2.0, 8.0),   # High syntactic complexity
        "rarity_range": (0.2, 0.8),  # Medium-high lexical rarity
        "cost_range": (0.02, 0.06),  # Higher cognitive cost
    },
    "fiction": {
        "vocab": [
            "she", "walked", "through", "the", "garden", "shadows", "dancing",
            "across", "stone", "walls", "whispered", "secrets", "ancient",
            "trees", "remembered", "everything", "heart", "pounding", "soft",
            "light", "fell", "like", "rain", "upon", "forgotten", "dreams",
            "silence", "broken", "only", "by", "wind", "rustling", "leaves",
        ],
        "depth_range": (1.0, 6.0),   # Varied, often simpler
        "rarity_range": (0.1, 0.6),  # More common words
        "cost_range": (0.01, 0.04),  # Lower cognitive cost
    },
    "journalism": {
        "vocab": [
            "officials", "confirmed", "yesterday", "according", "sources",
            "reported", "investigation", "revealed", "statement", "public",
            "government", "policy", "announced", "crisis", "response",
            "impact", "economic", "growth", "declined", "percent", "year",
            "experts", "warned", "significant", "development", "following",
        ],
        "depth_range": (1.5, 5.0),   # Clear, direct
        "rarity_range": (0.1, 0.5),  # Common vocabulary
        "cost_range": (0.01, 0.03),  # Lower cost (formulaic)
    },
    "technical": {
        "vocab": [
            "function", "returns", "parameter", "implementation", "algorithm",
            "complexity", "runtime", "memory", "allocated", "pointer",
            "interface", "protocol", "buffer", "thread", "mutex", "lock",
            "initialize", "configuration", "handler", "callback", "async",
            "pipeline", "deployment", "container", "service", "endpoint",
        ],
        "depth_range": (2.0, 7.0),   # High precision needed
        "rarity_range": (0.3, 0.9),  # Domain-specific (rare for general)
        "cost_range": (0.02, 0.05),  # Medium cost
    },
}


def _generate_domain_trace(
    domain: str,
    n_events: int,
    seed: int,
) -> List[CausalEvent]:
    """Generate an authentic trace with domain-specific cognitive profile."""
    cfg = DOMAINS[domain]
    rng = random.Random(seed)

    author = EmbodiedScholar(f"{domain}_author_{seed}", initial_glucose=rng.uniform(0.9, 1.0))
    engine = IrreversibleProcessEngine(author)

    vocab = cfg["vocab"]
    for i in range(n_events):
        word = vocab[i % len(vocab)]
        depth = rng.uniform(*cfg["depth_range"])
        rarity = rng.uniform(*cfg["rarity_range"])
        cost = rng.uniform(*cfg["cost_range"])
        engine.execute(LexicalIntention(word, depth, rarity, cost))

    return [
        CausalEvent(
            intention=evt.intention.target,
            actual_output=evt.actual_output,
            status="repair" if evt.failure_mode else "success",
            failure_mode=evt.failure_mode,
            repair_artifact=evt.actual_output if evt.failure_mode else None,
            glucose_at_event=evt.glucose_before,
            latency_ms=evt.latency_ms,
            syntactic_complexity=evt.intention.syntactic_depth,
        )
        for evt in engine.trace
    ]


def run_domain_transfer(
    n_samples: int = 50,
    n_events: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test detection transfer across domains."""
    sample_text = " ".join(_ACADEMIC_WORDS * 5)
    results: Dict[str, Any] = {"n_samples": n_samples, "n_events": n_events}

    for domain in DOMAINS:
        # Generate authentic traces in this domain
        authentic = [
            _generate_domain_trace(domain, n_events, seed + i)
            for i in range(n_samples)
        ]

        domain_results: Dict[str, Any] = {}

        for tier in ALL_TIERS:
            tier_seeds = {"naive": 1000, "statistical": 2000, "reverse_engineered": 3000, "expert": 4000}
            offset = tier_seeds.get(tier, 5000)

            forged = []
            for i in range(n_samples):
                gen = ForgedTraceGenerator(seed=seed + offset + i)
                if tier == "naive":
                    forged.append(gen.generate_naive_forgery(sample_text, n_events=n_events))
                elif tier == "statistical":
                    template = authentic[i % len(authentic)]
                    cs_trace = [
                        CausalEvent(
                            intention=e.intention, actual_output=e.actual_output,
                            status=e.status, failure_mode=e.failure_mode,
                            repair_artifact=e.repair_artifact,
                            glucose_at_event=e.glucose_at_event,
                            latency_ms=e.latency_ms,
                            syntactic_complexity=e.syntactic_complexity,
                        )
                        for e in template
                    ]
                    forged.append(gen.generate_statistical_forgery(sample_text, cs_trace))
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

            auth_sigs = [_extract_consciousness_signals(t) for t in authentic]
            forge_sigs = [_extract_consciousness_signals(t) for t in forged]
            y_true = [1.0] * n_samples + [0.0] * n_samples
            all_sigs = auth_sigs + forge_sigs
            scores = [sum(s.values()) / len(s) for s in all_sigs]
            pt, lo, hi = bootstrap_auc_ci(y_true, scores)

            domain_results[tier] = {"auc": pt, "ci": [lo, hi]}

        results[domain] = domain_results

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted cross-domain results."""
    print(f"\n{'=' * 90}")
    print(f"  Cross-Domain Transfer (n={results['n_samples']}, events={results['n_events']})")
    print(f"{'=' * 90}")
    print(f"{'Domain':<15}", end="")
    for tier in ALL_TIERS:
        print(f" {tier:>20}", end="")
    print()
    print("-" * 90)

    for domain in DOMAINS:
        print(f"{domain:<15}", end="")
        for tier in ALL_TIERS:
            d = results[domain][tier]
            print(f" {d['auc']:.3f} ({d['ci'][0]:.3f}-{d['ci'][1]:.3f}):>20", end="")
        print()
    print()


def _write_latex_table(results: Dict[str, Any], path: str) -> None:
    """Write booktabs LaTeX table."""
    n_tiers = len(ALL_TIERS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-domain generalization: AUC (95\% CI) by domain and adversary tier. "
        r"Detection is trained on academic writing patterns.}",
        r"\label{tab:domain-transfer}",
        r"\begin{tabular}{l" + "c" * n_tiers + "}",
        r"\toprule",
    ]

    header = "Domain"
    for t in ALL_TIERS:
        header += f" & {t.replace('_', ' ').title()}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for domain in DOMAINS:
        row = domain.title()
        for tier in ALL_TIERS:
            d = results[domain][tier]
            cell = f"{d['auc']:.3f} ({d['ci'][0]:.3f}--{d['ci'][1]:.3f})"
            row += f" & {cell}"
        row += r" \\"
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation")
    parser.add_argument("-n", "--n-samples", type=int, default=50)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("domain_transfer.json"))
    parser.add_argument("--output-latex", type=str, default=None)
    args = parser.parse_args()

    results = run_domain_transfer(args.n_samples, args.n_events, args.seed)

    # Print
    print(f"\n{'=' * 100}")
    print(f"  Cross-Domain Transfer (n={results['n_samples']}, events={results['n_events']})")
    print(f"{'=' * 100}")
    header = f"{'Domain':<15}"
    for tier in ALL_TIERS:
        header += f" {tier:>22}"
    print(header)
    print("-" * 100)
    for domain in DOMAINS:
        row = f"{domain:<15}"
        for tier in ALL_TIERS:
            d = results[domain][tier]
            row += f" {d['auc']:.3f} ({d['ci'][0]:.3f}-{d['ci'][1]:.3f}):>22"
        print(row)
    print()

    if args.output_latex:
        _write_latex_table(results, args.output_latex)
        print(f"LaTeX table: {args.output_latex}")

    args.output.write_text(json.dumps(results, indent=2, default=float))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
