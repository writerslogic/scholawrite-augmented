#!/usr/bin/env python
"""Privacy and membership inference analysis on causal traces.

Evaluates whether causal traces leak author identity by testing:
1. Author re-identification (given a trace, identify which author wrote it)
2. Membership inference (given a trace, was this author in training set?)

The first quantifies privacy risk. The second measures whether traces
contain enough author-specific information to enable membership attacks.

Usage:
    uv run python scripts/run_privacy_analysis.py -n 10
    uv run python scripts/run_privacy_analysis.py -n 20 --output-latex results/privacy.tex
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from scholawrite.causal_core import IrreversibleProcessEngine, LexicalIntention
from scholawrite.embodied import EmbodiedScholar
from scholawrite.schema import CausalEvent
from scholawrite.metrics import compute_causal_signatures, bootstrap_auc_ci, auc
from scholawrite.adversarial import _extract_consciousness_signals
from scholawrite.cross_author import _generate_diverse_authors


_TOKENS = [
    "however", "the", "analysis", "demonstrates", "that", "underlying",
    "mechanism", "provides", "novel", "framework", "for", "understanding",
    "complex", "interactions", "between", "cognitive", "processes", "and",
    "behavioral", "outcomes", "furthermore", "empirical", "evidence",
    "suggests", "paradigm", "shift", "methodology", "theoretical",
]


def _generate_author_traces(
    author_config: Dict[str, float],
    author_id: str,
    n_traces: int,
    n_events: int,
    seed: int,
) -> List[List[Dict[str, float]]]:
    """Generate multiple traces for one author, extract signatures."""
    rng = random.Random(seed)
    traces_sigs = []

    for t in range(n_traces):
        author = EmbodiedScholar(
            author_id,
            initial_glucose=author_config["initial_glucose"],
        )
        engine = IrreversibleProcessEngine(author)

        for j in range(n_events):
            word = _TOKENS[(t * n_events + j) % len(_TOKENS)]
            depth = rng.uniform(1.0, 7.0)
            rarity = rng.uniform(0.1, 0.7)
            cost = rng.uniform(0.01, 0.05)
            engine.execute(LexicalIntention(word, depth, rarity, cost))

        # Convert ExecutionEvent -> CausalEvent for metrics
        causal_trace = [
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
        sigs = compute_causal_signatures(causal_trace)
        traces_sigs.append(sigs)

    return traces_sigs


def _signature_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Euclidean distance between two signature dicts."""
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in keys))


def run_reidentification(
    n_authors: int = 10,
    traces_per_author: int = 10,
    n_events: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test author re-identification from causal traces.

    For each author, uses half traces as reference (centroid), tests remaining.
    """
    configs = _generate_diverse_authors(n_authors, seed)

    all_author_sigs: List[Tuple[str, List[Dict[str, float]]]] = []
    for i, cfg in enumerate(configs):
        sigs = _generate_author_traces(cfg, f"author_{i}", traces_per_author, n_events, seed + i * 100)
        all_author_sigs.append((f"author_{i}", sigs))

    # Split each author's traces into reference (first half) and test (second half)
    half = traces_per_author // 2
    centroids: Dict[str, Dict[str, float]] = {}
    for author_id, sigs in all_author_sigs:
        ref = sigs[:half]
        keys = ref[0].keys()
        centroids[author_id] = {k: mean(s[k] for s in ref) for k in keys}

    # Re-identification: for each test trace, find nearest centroid
    correct = 0
    total = 0
    per_author_accuracy: Dict[str, float] = {}

    for author_id, sigs in all_author_sigs:
        test = sigs[half:]
        author_correct = 0
        for sig in test:
            distances = {
                aid: _signature_distance(sig, cent)
                for aid, cent in centroids.items()
            }
            predicted = min(distances, key=distances.get)
            if predicted == author_id:
                correct += 1
                author_correct += 1
            total += 1
        per_author_accuracy[author_id] = round(author_correct / len(test), 4) if test else 0.0

    overall_accuracy = correct / total if total else 0.0
    chance_level = 1.0 / n_authors

    return {
        "n_authors": n_authors,
        "traces_per_author": traces_per_author,
        "n_events": n_events,
        "overall_accuracy": round(overall_accuracy, 4),
        "chance_level": round(chance_level, 4),
        "privacy_risk": round(overall_accuracy / chance_level, 2),  # ratio above chance
        "per_author_accuracy": per_author_accuracy,
    }


def run_membership_inference(
    n_authors_train: int = 10,
    n_authors_test: int = 10,
    traces_per_author: int = 10,
    n_events: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test membership inference: can we tell if an author was in training set?

    Trains on n_authors_train, tests with their traces (members) and
    traces from n_authors_test unseen authors (non-members).
    """
    configs_train = _generate_diverse_authors(n_authors_train, seed)
    configs_test = _generate_diverse_authors(n_authors_test, seed + 10000)

    # Generate training centroids
    train_sigs = []
    for i, cfg in enumerate(configs_train):
        sigs = _generate_author_traces(cfg, f"train_{i}", traces_per_author, n_events, seed + i * 100)
        train_sigs.append(sigs)

    # Build per-author centroids from training traces
    centroids = []
    for sigs in train_sigs:
        keys = sigs[0].keys()
        centroids.append({k: mean(s[k] for s in sigs) for k in keys})

    # Grand centroid (average of all training authors)
    grand_keys = centroids[0].keys()
    grand_centroid = {k: mean(c[k] for c in centroids) for k in grand_keys}

    # For members: generate new traces, compute distance to nearest centroid
    member_scores = []
    for i, cfg in enumerate(configs_train):
        new_sigs = _generate_author_traces(cfg, f"train_{i}", 3, n_events, seed + 50000 + i * 100)
        for sig in new_sigs:
            min_dist = min(_signature_distance(sig, c) for c in centroids)
            member_scores.append(1.0 - min(1.0, min_dist))  # higher = closer = member

    # For non-members: generate traces, compute distance to nearest centroid
    nonmember_scores = []
    for i, cfg in enumerate(configs_test):
        new_sigs = _generate_author_traces(cfg, f"test_{i}", 3, n_events, seed + 60000 + i * 100)
        for sig in new_sigs:
            min_dist = min(_signature_distance(sig, c) for c in centroids)
            nonmember_scores.append(1.0 - min(1.0, min_dist))

    # AUC for membership inference (members = positive class)
    y_true = [1.0] * len(member_scores) + [0.0] * len(nonmember_scores)
    all_scores = member_scores + nonmember_scores
    mi_auc = auc(y_true, all_scores)
    pt, lo, hi = bootstrap_auc_ci(y_true, all_scores)

    return {
        "n_authors_train": n_authors_train,
        "n_authors_test": n_authors_test,
        "membership_inference_auc": {"auc": pt, "ci": [lo, hi]},
        "mean_member_score": round(mean(member_scores), 4),
        "mean_nonmember_score": round(mean(nonmember_scores), 4),
        "privacy_assessment": (
            "low_risk" if mi_auc < 0.6
            else "moderate_risk" if mi_auc < 0.75
            else "high_risk"
        ),
    }


def print_results(reid: Dict[str, Any], mi: Dict[str, Any]) -> None:
    """Print formatted results."""
    print(f"\n{'=' * 70}")
    print(f"  Privacy Analysis")
    print(f"{'=' * 70}")

    print(f"\n--- Author Re-identification ---")
    print(f"  Authors: {reid['n_authors']}, Traces/author: {reid['traces_per_author']}")
    print(f"  Overall accuracy: {reid['overall_accuracy']:.3f} (chance: {reid['chance_level']:.3f})")
    print(f"  Privacy risk ratio: {reid['privacy_risk']:.1f}x above chance")

    print(f"\n--- Membership Inference ---")
    mi_a = mi["membership_inference_auc"]
    print(f"  AUC: {mi_a['auc']:.3f} ({mi_a['ci'][0]:.3f}-{mi_a['ci'][1]:.3f})")
    print(f"  Mean member score:     {mi['mean_member_score']:.4f}")
    print(f"  Mean non-member score: {mi['mean_nonmember_score']:.4f}")
    print(f"  Assessment: {mi['privacy_assessment']}")
    print()


def _write_latex_table(reid: Dict[str, Any], mi: Dict[str, Any], path: str) -> None:
    """Write booktabs LaTeX table."""
    mi_a = mi["membership_inference_auc"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Privacy analysis of causal execution traces. "
        r"Re-identification accuracy and membership inference AUC "
        r"quantify identity leakage risk.}",
        r"\label{tab:privacy-analysis}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Value & Baseline \\",
        r"\midrule",
        f"Re-ID accuracy & {reid['overall_accuracy']:.3f} & {reid['chance_level']:.3f} (chance)" + r" \\",
        f"Privacy risk ratio & {reid['privacy_risk']:.1f}$\\times$ & 1.0$\\times$" + r" \\",
        f"MI AUC (95\\% CI) & {mi_a['auc']:.3f} ({mi_a['ci'][0]:.3f}--{mi_a['ci'][1]:.3f}) & 0.500 (random)" + r" \\",
        f"Assessment & \\textit{{{mi['privacy_assessment'].replace('_', ' ')}}} &" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Privacy/membership inference analysis")
    parser.add_argument("-n", "--n-authors", type=int, default=10)
    parser.add_argument("--traces-per-author", type=int, default=10)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("privacy_analysis.json"))
    parser.add_argument("--output-latex", type=str, default=None)
    args = parser.parse_args()

    reid = run_reidentification(args.n_authors, args.traces_per_author, args.n_events, args.seed)
    mi = run_membership_inference(args.n_authors, args.n_authors, args.traces_per_author, args.n_events, args.seed)

    print_results(reid, mi)

    if args.output_latex:
        _write_latex_table(reid, mi, args.output_latex)
        print(f"LaTeX table: {args.output_latex}")

    combined = {"reidentification": reid, "membership_inference": mi}
    args.output.write_text(json.dumps(combined, indent=2, default=float))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
