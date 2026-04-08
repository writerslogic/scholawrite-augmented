#!/usr/bin/env python3
"""Compare real human-written texts against AI-generated texts using GPTZero and Originality.ai.

Loads human seed texts and AI-generated outputs (GPT-4o, Llama-meta, Llama-SW),
strips LaTeX markup, runs each through external detectors, and reports AUC.

Usage:
    uv run python scripts/run_real_text_comparison.py -o results/real_text_comparison.json

    # Dry run (no API calls, just verify file loading)
    uv run python scripts/run_real_text_comparison.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from scholawrite.detector_harness import (
    DetectorHarness,
    GptZeroDetector,
    OriginalityDetector,
)

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent  # scholawrite-augmented/
SEEDS_DIR = ROOT / "seeds"
OUTPUTS_DIR = ROOT / "outputs"

SEED_IDS = [1, 2, 3, 4]
ITER_SNAPSHOTS = [0, 25, 50, 75, 99]

MODEL_DIRS = {
    "gpt4o": "gpt4o_output",
    "llama8_meta": "llama8_meta_output",
    "llama8_SW": "llama8_SW_output",
}


# ── LaTeX stripping ───────────────────────────────────────────────────────

def strip_latex(text: str) -> str:
    """Strip LaTeX markup, keeping plain text content."""
    # Remove full-line comments
    text = re.sub(r"(?m)^%.*$", "", text)
    # Remove inline comments (unescaped %)
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    # Remove \documentclass, \usepackage, \maketitle, \label, \centering lines
    text = re.sub(r"\\(?:documentclass|usepackage|maketitle|label|centering|date|bibliography\w*)\b[^\n]*", "", text)
    # Remove \begin{...} and \end{...} tags (but keep content between them)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
    # Remove \includegraphics and similar commands with optional+required args
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    # Remove \cite{...}, \ref{...}, \label{...}
    text = re.sub(r"\\(?:cite|ref|label|eqref|pageref)\{[^}]*\}", "", text)
    # Remove \title{...}, \author{...} but keep braced content
    text = re.sub(r"\\(?:title|author|section|subsection|subsubsection|caption|footnote|textit|textbf|emph|text\w+)\{([^}]*)\}", r"\1", text)
    # Remove remaining \command (no braces)
    text = re.sub(r"\\(?:item|hline|toprule|midrule|bottomrule|noindent|newpage|clearpage|bigskip|medskip|smallskip|par)\b", "", text)
    # Remove \\ (line breaks)
    text = re.sub(r"\\\\", " ", text)
    # Remove remaining \command{content} keeping content
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # Remove remaining \command without braces
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove $...$ (inline math) keeping content
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    # Remove stray braces
    text = re.sub(r"[{}]", "", text)
    # Remove stray brackets from optional args
    text = re.sub(r"\[[^\]]*\]", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── File loading ─────────────────────────────────────────────────────────

def load_human_texts() -> Dict[str, str]:
    """Load and strip human seed texts. Returns {seed_id: stripped_text}."""
    texts = {}
    for sid in SEED_IDS:
        path = SEEDS_DIR / f"seed{sid}.txt"
        raw = path.read_text(encoding="utf-8", errors="replace")
        texts[f"seed{sid}"] = strip_latex(raw)
    return texts


def load_ai_texts() -> Dict[str, List[Tuple[str, str]]]:
    """Load AI-generated texts grouped by model.

    Returns {model_name: [(label, stripped_text), ...]}.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for model_name, dir_name in MODEL_DIRS.items():
        items: List[Tuple[str, str]] = []
        for sid in SEED_IDS:
            for it in ITER_SNAPSHOTS:
                path = OUTPUTS_DIR / dir_name / f"seed{sid}" / "generation" / f"iter_generation_{it}.txt"
                if not path.exists():
                    print(f"  WARNING: missing {path}", file=sys.stderr)
                    continue
                raw = path.read_text(encoding="utf-8", errors="replace")
                label = f"{model_name}/seed{sid}/iter{it}"
                items.append((label, strip_latex(raw)))
        groups[model_name] = items
    return groups


# ── Main ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> dict:
    human_texts = load_human_texts()
    ai_groups = load_ai_texts()

    print(f"Human texts: {len(human_texts)}")
    for k, v in human_texts.items():
        print(f"  {k}: {len(v)} chars")

    print(f"\nAI text groups: {len(ai_groups)}")
    for model, items in ai_groups.items():
        print(f"  {model}: {len(items)} texts")

    if args.dry_run:
        print("\n[dry run] Skipping API calls.")
        return {}

    # Build detectors
    detectors = [GptZeroDetector(), OriginalityDetector()]
    harness = DetectorHarness(detectors)

    human_list = list(human_texts.values())
    machine_by_tier = {model: [t for _, t in items] for model, items in ai_groups.items()}

    # Run harness
    print("\nRunning detectors (this may take a while due to rate limiting)...")
    harness_result = harness.evaluate_on_texts(human_list, machine_by_tier)

    # Build per-text detail
    print("\nScoring individual texts...")
    individual_scores: List[dict] = []

    for det in detectors:
        # Score human texts
        for label, text in human_texts.items():
            r = det.detect(text)
            individual_scores.append({
                "detector": det.name,
                "label": label,
                "source": "human",
                "score": r.score,
                "classification": "human" if r.score >= 0.5 else "ai",
                "error": r.error,
            })

        # Score AI texts
        for model, items in ai_groups.items():
            for label, text in items:
                r = det.detect(text)
                individual_scores.append({
                    "detector": det.name,
                    "label": label,
                    "source": model,
                    "score": r.score,
                    "classification": "human" if r.score >= 0.5 else "ai",
                    "error": r.error,
                })

    # Print summary table
    print("\n" + "=" * 72)
    print("DETECTOR PERFORMANCE (AUC: human vs AI model)")
    print("=" * 72)
    matrix = harness_result.get("matrix", {})
    tiers = harness_result.get("tiers", [])

    # Header
    header = f"{'Detector':<20}" + "".join(f"{t:<16}" for t in tiers)
    print(header)
    print("-" * len(header))
    for det_name, tier_aucs in matrix.items():
        row = f"{det_name:<20}"
        for tier in tiers:
            val = tier_aucs.get(tier, -1.0)
            row += f"{val:<16.4f}"
        print(row)

    # Print detail
    print("\n" + "=" * 72)
    print("PER-DETECTOR DETAIL")
    print("=" * 72)
    detail = harness_result.get("detail", {})
    for det_name, tier_detail in detail.items():
        print(f"\n  {det_name}:")
        for tier, d in tier_detail.items():
            print(f"    vs {tier}: AUC={d['auc']:.4f}  "
                  f"human_mean={d['human_mean']:.4f}  "
                  f"machine_mean={d['machine_mean']:.4f}  "
                  f"n_human={d['n_human']}  n_machine={d['n_machine']}  "
                  f"errors={d['errors']}")

    # Print individual misclassifications
    print("\n" + "=" * 72)
    print("INDIVIDUAL TEXT SCORES")
    print("=" * 72)
    print(f"{'Detector':<18}{'Source':<14}{'Label':<35}{'Score':>6}  {'Class'}")
    print("-" * 85)
    for entry in individual_scores:
        print(f"{entry['detector']:<18}{entry['source']:<14}{entry['label']:<35}"
              f"{entry['score']:>6.4f}  {entry['classification']}"
              + (f"  ERR:{entry['error']}" if entry['error'] else ""))

    # Assemble output
    output = {
        "harness": harness_result,
        "individual_scores": individual_scores,
    }

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare human vs AI texts using GPTZero and Originality.ai"
    )
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load files only, skip API calls")
    args = parser.parse_args()

    result = run(args)

    if args.output and result:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
