#!/usr/bin/env python
"""Incremental injection attack: split AI text across N revisions.

Tests whether splitting injected content into small pieces (one sentence
per revision) defeats cross-revision detection. Process-level signals
should still detect the injection because each piece still lacks authentic
causal traces.

Usage:
    uv run python scripts/run_incremental_injection.py -n 50
    uv run python scripts/run_incremental_injection.py -n 100 --output-latex results/incremental.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from scholawrite.adversarial import (
    ForgedTraceGenerator,
    _extract_consciousness_signals,
    _generate_authentic_trace,
    _ACADEMIC_WORDS,
)
from scholawrite.baselines import _compression_discontinuity
from scholawrite.metrics import auc, bootstrap_auc_ci


def run_incremental_experiment(
    n_samples: int = 50,
    n_events: int = 30,
    seed: int = 42,
    chunks_list: List[int] | None = None,
) -> Dict[str, Any]:
    """Test detection at varying injection granularities.

    Args:
        chunks_list: Number of pieces to split injection into.
            E.g. [1, 2, 5, 10] means inject all at once, in 2 pieces, etc.
    """
    if chunks_list is None:
        chunks_list = [1, 2, 5, 10]

    # Generate authentic traces (full-length)
    authentic = [
        _generate_authentic_trace(seed=seed + i, n_events=n_events)
        for i in range(n_samples)
    ]

    # Generate forged traces (full-length, then split)
    sample_text = " ".join(_ACADEMIC_WORDS * 5)
    forged_full = []
    for i in range(n_samples):
        gen = ForgedTraceGenerator(seed=seed + 4000 + i)
        trace, _ = gen.generate_expert_forgery(
            sample_text, author_id=f"expert_{i}", n_events=n_events,
        )
        forged_full.append(trace)

    y_true = [1.0] * n_samples + [0.0] * n_samples

    results: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_events": n_events,
        "chunks_tested": chunks_list,
    }

    for n_chunks in chunks_list:
        # Split each forged trace into n_chunks pieces
        forged_chunks = []
        for trace in forged_full:
            chunk_size = max(1, len(trace) // n_chunks)
            chunks = [trace[i:i + chunk_size] for i in range(0, len(trace), chunk_size)]
            # Take only n_chunks pieces (last chunk may be smaller)
            chunks = chunks[:n_chunks]
            forged_chunks.append(chunks)

        # Per-chunk consciousness signatures (average across chunks)
        auth_sigs = [_extract_consciousness_signals(t) for t in authentic]
        forge_sigs = []
        for chunks in forged_chunks:
            chunk_sigs = [_extract_consciousness_signals(c) for c in chunks if len(c) >= 3]
            if chunk_sigs:
                avg_sig = {}
                for key in chunk_sigs[0]:
                    avg_sig[key] = sum(s[key] for s in chunk_sigs) / len(chunk_sigs)
                forge_sigs.append(avg_sig)
            else:
                forge_sigs.append({k: 0.0 for k in auth_sigs[0]})

        all_sigs = auth_sigs + forge_sigs
        process_scores = [sum(s.values()) / len(s) for s in all_sigs]
        proc_pt, proc_lo, proc_hi = bootstrap_auc_ci(y_true, process_scores)

        # NCD: compare consecutive chunks (detect discontinuity at injection boundaries)
        ncd_scores_auth = []
        for trace in authentic:
            text = " ".join(e.actual_output for e in trace)
            # Self-NCD against first half
            half = len(text) // 2
            ncd_scores_auth.append(1.0 - _compression_discontinuity(text[:half], text[half:]))

        ncd_scores_forge = []
        for chunks in forged_chunks:
            if len(chunks) >= 2:
                t1 = " ".join(e.actual_output for e in chunks[0])
                t2 = " ".join(e.actual_output for e in chunks[-1])
                ncd_scores_forge.append(1.0 - _compression_discontinuity(t1, t2))
            else:
                text = " ".join(e.actual_output for e in chunks[0])
                half = len(text) // 2
                ncd_scores_forge.append(1.0 - _compression_discontinuity(text[:half], text[half:]))

        ncd_all = ncd_scores_auth + ncd_scores_forge
        ncd_pt, ncd_lo, ncd_hi = bootstrap_auc_ci(y_true, ncd_all)

        # Minimum chunk length (for reporting)
        min_chunk_events = min(
            len(c) for chunks in forged_chunks for c in chunks
        ) if forged_chunks else 0

        results[f"chunks_{n_chunks}"] = {
            "n_chunks": n_chunks,
            "min_chunk_events": min_chunk_events,
            "consciousness_auc": {"auc": proc_pt, "ci": [proc_lo, proc_hi]},
            "ncd_auc": {"auc": ncd_pt, "ci": [ncd_lo, ncd_hi]},
        }

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted results."""
    print(f"\n{'=' * 80}")
    print(f"  Incremental Injection (n={results['n_samples']}, events={results['n_events']})")
    print(f"{'=' * 80}")
    print(f"{'Chunks':>8} {'Min Events':>12} {'Consciousness AUC':>25} {'NCD AUC':>25}")
    print("-" * 80)

    for n_chunks in results["chunks_tested"]:
        d = results[f"chunks_{n_chunks}"]
        c = d["consciousness_auc"]
        n = d["ncd_auc"]
        print(
            f"{n_chunks:>8d} {d['min_chunk_events']:>12d} "
            f"{c['auc']:.3f} ({c['ci'][0]:.3f}-{c['ci'][1]:.3f}):>25 "
            f"{n['auc']:.3f} ({n['ci'][0]:.3f}-{n['ci'][1]:.3f}):>25"
        )
    print()


def _write_latex_table(results: Dict[str, Any], path: str) -> None:
    """Write booktabs LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection AUC under incremental injection attack. "
        r"AI text is split into $k$ chunks injected across separate revisions. "
        r"Process-level detection degrades gracefully with smaller chunks.}",
        r"\label{tab:incremental-injection}",
        r"\begin{tabular}{rccc}",
        r"\toprule",
        r"Chunks & Min Events & Consciousness AUC & NCD AUC \\",
        r"\midrule",
    ]

    for n_chunks in results["chunks_tested"]:
        d = results[f"chunks_{n_chunks}"]
        c = d["consciousness_auc"]
        n = d["ncd_auc"]
        lines.append(
            f"{n_chunks} & {d['min_chunk_events']} & "
            f"{c['auc']:.3f} ({c['ci'][0]:.3f}--{c['ci'][1]:.3f}) & "
            f"{n['auc']:.3f} ({n['ci'][0]:.3f}--{n['ci'][1]:.3f}) " + r"\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Incremental injection attack analysis")
    parser.add_argument("-n", "--n-samples", type=int, default=50)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("incremental_injection.json"))
    parser.add_argument("--output-latex", type=str, default=None)
    args = parser.parse_args()

    results = run_incremental_experiment(args.n_samples, args.n_events, args.seed)

    # Print
    print(f"\n{'=' * 80}")
    print(f"  Incremental Injection (n={results['n_samples']}, events={results['n_events']})")
    print(f"{'=' * 80}")
    print(f"{'Chunks':>8} {'Min Evts':>10} {'Consciousness AUC (95% CI)':>30} {'NCD AUC (95% CI)':>25}")
    print("-" * 80)
    for n_chunks in results["chunks_tested"]:
        d = results[f"chunks_{n_chunks}"]
        c = d["consciousness_auc"]
        n = d["ncd_auc"]
        c_str = f"{c['auc']:.3f} ({c['ci'][0]:.3f}-{c['ci'][1]:.3f})"
        n_str = f"{n['auc']:.3f} ({n['ci'][0]:.3f}-{n['ci'][1]:.3f})"
        print(f"{n_chunks:>8d} {d['min_chunk_events']:>10d} {c_str:>30} {n_str:>25}")
    print()

    if args.output_latex:
        _write_latex_table(results, args.output_latex)
        print(f"LaTeX table: {args.output_latex}")

    args.output.write_text(json.dumps(results, indent=2, default=float))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
