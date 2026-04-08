#!/usr/bin/env python
"""Paraphrase robustness analysis for process vs product-level detection.

Tests whether detection degrades when injected text is paraphrased.
Process-level signals (consciousness signatures) operate on causal traces,
not text, so they should be invariant. Text-level baselines (NCD, perplexity,
burstiness) operate on the output text and may degrade.

Usage:
    uv run python scripts/run_paraphrase_robustness.py -n 50
    uv run python scripts/run_paraphrase_robustness.py -n 100 --output-latex results/paraphrase.tex
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scholawrite.adversarial import (
    AdversarialEvaluator,
    ForgedTraceGenerator,
    _extract_consciousness_signals,
    _generate_authentic_trace,
    _ACADEMIC_WORDS,
)
from scholawrite.baselines import (
    perplexity_proxy_score,
    ngram_burstiness_score,
    _compression_discontinuity,
)
from scholawrite.metrics import auc, bootstrap_auc_ci


# ── Rule-based paraphrase transforms ─────────────────────────────────────

_SYNONYMS = {
    "important": "significant", "significant": "important",
    "demonstrate": "show", "show": "demonstrate",
    "utilize": "use", "use": "employ",
    "however": "nevertheless", "nevertheless": "however",
    "therefore": "consequently", "consequently": "therefore",
    "approach": "method", "method": "approach",
    "result": "outcome", "outcome": "result",
    "analyze": "examine", "examine": "analyze",
    "suggest": "indicate", "indicate": "suggest",
    "improve": "enhance", "enhance": "improve",
    "provide": "offer", "offer": "provide",
    "require": "need", "need": "require",
    "obtain": "acquire", "acquire": "obtain",
    "increase": "raise", "raise": "elevate",
    "decrease": "reduce", "reduce": "lower",
    "previous": "prior", "prior": "earlier",
    "additional": "further", "further": "extra",
}


def _synonym_substitution(text: str, rng: random.Random, rate: float = 0.3) -> str:
    """Replace words with synonyms at the given rate."""
    words = text.split()
    result = []
    for w in words:
        lower = w.lower().strip(".,;:!?")
        if lower in _SYNONYMS and rng.random() < rate:
            replacement = _SYNONYMS[lower]
            # Preserve case
            if w[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            trailing = ""
            for c in reversed(w):
                if c in ".,;:!?":
                    trailing = c + trailing
                else:
                    break
            result.append(replacement + trailing)
        else:
            result.append(w)
    return " ".join(result)


def _sentence_shuffle(text: str, rng: random.Random) -> str:
    """Shuffle sentence order (preserving first and last)."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sents) <= 3:
        return text
    middle = sents[1:-1]
    rng.shuffle(middle)
    return " ".join([sents[0]] + middle + [sents[-1]])


def _word_insertion(text: str, rng: random.Random, rate: float = 0.1) -> str:
    """Insert filler words at random positions."""
    fillers = ["indeed", "notably", "specifically", "essentially", "particularly"]
    words = text.split()
    result = []
    for w in words:
        if rng.random() < rate:
            result.append(rng.choice(fillers))
        result.append(w)
    return " ".join(result)


def paraphrase(text: str, rng: random.Random, intensity: str = "light") -> str:
    """Apply rule-based paraphrase transforms."""
    if intensity == "light":
        return _synonym_substitution(text, rng, rate=0.2)
    elif intensity == "moderate":
        text = _synonym_substitution(text, rng, rate=0.4)
        text = _word_insertion(text, rng, rate=0.05)
        return text
    else:  # heavy
        text = _synonym_substitution(text, rng, rate=0.5)
        text = _sentence_shuffle(text, rng)
        text = _word_insertion(text, rng, rate=0.1)
        return text


# ── Experiment ────────────────────────────────────────────────────────────

def run_paraphrase_experiment(
    n_samples: int = 50,
    n_events: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run paraphrase robustness analysis across intensities."""
    rng = random.Random(seed)

    # Generate authentic + forged traces
    authentic_traces = [
        _generate_authentic_trace(seed=seed + i, n_events=n_events)
        for i in range(n_samples)
    ]
    sample_text = " ".join(_ACADEMIC_WORDS * 5)
    forged_traces = []
    for i in range(n_samples):
        gen = ForgedTraceGenerator(seed=seed + 4000 + i)
        trace, _ = gen.generate_expert_forgery(
            sample_text, author_id=f"expert_{i}", n_events=n_events,
        )
        forged_traces.append(trace)

    # Generate text samples (use actual_output from traces)
    def _trace_text(trace: list) -> str:
        return " ".join(e.actual_output for e in trace)

    auth_texts = [_trace_text(t) for t in authentic_traces]
    forge_texts = [_trace_text(t) for t in forged_traces]
    y_true = [1.0] * n_samples + [0.0] * n_samples

    results: Dict[str, Any] = {"n_samples": n_samples, "n_events": n_events}

    for intensity in ["none", "light", "moderate", "heavy"]:
        # Paraphrase forged texts (authentic stay untouched)
        if intensity == "none":
            para_forge = forge_texts
        else:
            para_forge = [paraphrase(t, random.Random(seed + i), intensity) for i, t in enumerate(forge_texts)]

        all_texts = auth_texts + para_forge

        # 1. Process-level: consciousness signatures (trace-based, text-independent)
        auth_sigs = [_extract_consciousness_signals(t) for t in authentic_traces]
        forge_sigs = [_extract_consciousness_signals(t) for t in forged_traces]
        all_sigs = auth_sigs + forge_sigs
        process_scores = [sum(s.values()) / len(s) for s in all_sigs]
        proc_pt, proc_lo, proc_hi = bootstrap_auc_ci(y_true, process_scores)

        # 2. Text-level: perplexity proxy
        ppl_scores = [perplexity_proxy_score(t) for t in all_texts]
        ppl_pt, ppl_lo, ppl_hi = bootstrap_auc_ci(y_true, ppl_scores)

        # 3. Text-level: n-gram burstiness
        burst_scores = [ngram_burstiness_score(t) for t in all_texts]
        burst_pt, burst_lo, burst_hi = bootstrap_auc_ci(y_true, burst_scores)

        # 4. Text-level: NCD vs reference
        ref_text = " ".join(_ACADEMIC_WORDS * 3)
        ncd_scores = [1.0 - _compression_discontinuity(ref_text, t) for t in all_texts]
        ncd_pt, ncd_lo, ncd_hi = bootstrap_auc_ci(y_true, ncd_scores)

        results[intensity] = {
            "process_consciousness": {"auc": proc_pt, "ci": [proc_lo, proc_hi]},
            "perplexity_proxy": {"auc": ppl_pt, "ci": [ppl_lo, ppl_hi]},
            "ngram_burstiness": {"auc": burst_pt, "ci": [burst_lo, burst_hi]},
            "ncd": {"auc": ncd_pt, "ci": [ncd_lo, ncd_hi]},
        }

    # Compute degradation (delta from "none" baseline)
    for intensity in ["light", "moderate", "heavy"]:
        for detector in ["process_consciousness", "perplexity_proxy", "ngram_burstiness", "ncd"]:
            baseline_auc = results["none"][detector]["auc"]
            current_auc = results[intensity][detector]["auc"]
            results[intensity][detector]["delta"] = round(current_auc - baseline_auc, 4)

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted results table."""
    detectors = ["process_consciousness", "perplexity_proxy", "ngram_burstiness", "ncd"]
    intensities = ["none", "light", "moderate", "heavy"]

    print(f"\n{'=' * 90}")
    print(f"  Paraphrase Robustness (n={results['n_samples']}, events={results['n_events']})")
    print(f"{'=' * 90}")
    header = f"{'Detector':<25}"
    for i in intensities:
        header += f" {i:>14}"
    print(header)
    print("-" * 90)

    for det in detectors:
        row = f"{det:<25}"
        for i in intensities:
            d = results[i][det]
            val = f"{d['auc']:.3f}"
            if "delta" in d and d["delta"] != 0:
                val += f"({d['delta']:+.3f})"
            row += f" {val:>14}"
        print(row)
    print()


def _write_latex_table(results: Dict[str, Any], path: str) -> None:
    """Write booktabs LaTeX table."""
    import os
    detectors = ["process_consciousness", "perplexity_proxy", "ngram_burstiness", "ncd"]
    det_labels = {
        "process_consciousness": "Consciousness Sigs.",
        "perplexity_proxy": "Perplexity Proxy",
        "ngram_burstiness": "N-gram Burstiness",
        "ncd": "NCD",
    }
    intensities = ["none", "light", "moderate", "heavy"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection AUC under paraphrase attack at increasing intensity. "
        r"Process-level signals (consciousness signatures) are invariant to text "
        r"modification as they operate on causal traces.}",
        r"\label{tab:paraphrase-robustness}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Detector & None & Light & Moderate & Heavy \\",
        r"\midrule",
    ]

    for det in detectors:
        label = det_labels[det]
        cells = [label]
        for i in intensities:
            d = results[i][det]
            pt, lo, hi = d["auc"], d["ci"][0], d["ci"][1]
            cell = f"{pt:.3f} ({lo:.3f}--{hi:.3f})"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Paraphrase robustness analysis")
    parser.add_argument("-n", "--n-samples", type=int, default=50)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("paraphrase_robustness.json"))
    parser.add_argument("--output-latex", type=str, default=None)
    args = parser.parse_args()

    results = run_paraphrase_experiment(args.n_samples, args.n_events, args.seed)
    print_results(results)

    if args.output_latex:
        _write_latex_table(results, args.output_latex)
        print(f"LaTeX table: {args.output_latex}")

    args.output.write_text(json.dumps(results, indent=2, default=float))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
