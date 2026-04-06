#!/usr/bin/env python
"""Validate embodied simulation against real keystroke recorder data.

Addresses SYS-002/C-006: all traces are synthetic with no real human validation.
Compares checkpoint analytics from keystroke-recorder against EmbodiedScholar
simulation via KS tests, Mann-Whitney U, effect sizes, and Wasserstein distance.

Usage:
    uv run python scripts/validate_keystroke.py --quick
    uv run python scripts/validate_keystroke.py --output results/validation.json --output-latex results/validation.tex
    uv run python scripts/validate_keystroke.py --db ~/.local/data/keystrokes.db
    uv run python scripts/validate_keystroke.py --app-filter TextEdit
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def _out(*args: object, **kwargs: object) -> None:
    """Write to stdout (CLI output, not debug logging)."""
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


from scholawrite.validation import (
    SessionSummary,
    SignalComparison,
    ValidationResult,
    run_validation,
)


def print_session_table(sessions: list[SessionSummary]) -> None:
    _out("\n" + "=" * 95)
    _out("  QUALIFYING SESSIONS")
    _out("=" * 95)
    header = f"  {'Session ID':<38} {'App':<22} {'Keys':>6} {'WPM':>6} {'Chkpts':>7} {'Chain':>6} {'Min':>6}"
    _out(header)
    _out("-" * 95)
    for s in sessions:
        chain_str = "OK" if s.chain_intact else "BROKEN"
        _out(
            f"  {s.session_id:<38} {s.primary_app[:20]:<22} "
            f"{s.total_keystrokes:>6} {s.avg_wpm:>6.1f} "
            f"{s.n_active_checkpoints:>7} {chain_str:>6} {s.duration_minutes:>6.1f}"
        )
    _out("-" * 95)
    _out(f"  Total: {len(sessions)} sessions, "
          f"{sum(s.n_active_checkpoints for s in sessions)} active checkpoints, "
          f"{sum(s.chain_links_verified for s in sessions)} hash links verified")


def print_signal_table(comparisons: list[SignalComparison]) -> None:
    _out("\n" + "=" * 110)
    _out("  SIGNAL DISTRIBUTION COMPARISON: Real Keystrokes vs EmbodiedScholar Simulation")
    _out("=" * 110)
    header = (
        f"  {'Signal':<22} {'Real Mdn':>9} {'Sim Mdn':>9} "
        f"{'KS D':>7} {'KS p':>7} {'Cohen d':>8} {'Cliff d':>8} {'W1':>9}"
    )
    _out(header)
    _out("-" * 110)
    for c in comparisons:
        sig = "*" if c.ks_pvalue < 0.05 else " "
        _out(
            f"  {c.signal_name:<22} {c.real_median:>9.3f} {c.sim_median:>9.3f} "
            f"{c.ks_statistic:>7.3f} {c.ks_pvalue:>6.3f}{sig} "
            f"{c.cohens_d:>8.3f} {c.cliffs_delta:>8.3f} {c.wasserstein_distance:>9.3f}"
        )
    _out("-" * 110)
    _out("  * p < 0.05 (distributions differ significantly)")
    _out(f"  Real: n={comparisons[0].real_n if comparisons else 0} checkpoints  "
          f"Simulated: n={comparisons[0].sim_n if comparisons else 0} traces")

    # Effect size interpretation
    _out("\n  Effect size guide: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large")


def print_phase_table(phase: dict) -> None:
    _out("\n" + "=" * 60)
    _out("  FLOWER & HAYES (1981) THREE-PHASE PROPORTIONS")
    _out("=" * 60)
    real = phase["real"]
    sim = phase["simulated"]
    _out(f"  {'Phase':<25} {'Real (%)':>10} {'Simulated (%)':>14}")
    _out("-" * 60)
    for phase_name in ["planning", "translating", "revising"]:
        _out(f"  {phase_name.capitalize():<25} {real[phase_name]*100:>9.1f}% {sim[phase_name]*100:>13.1f}%")
    _out("-" * 60)
    _out(f"  Real events: {phase['real_total_events']}  |  Sim traces: {phase['sim_n_traces']}")


def print_chain_integrity(chain: dict) -> None:
    _out("\n" + "=" * 60)
    _out("  HASH CHAIN INTEGRITY (Observational Privilege)")
    _out("=" * 60)
    status = "INTACT" if chain["all_intact"] else "BROKEN"
    _out(f"  Status: {status}")
    _out(f"  Sessions checked: {chain['sessions_checked']}")
    _out(f"  Total links verified: {chain['total_links_verified']}")
    if not chain["all_intact"]:
        for sid, info in chain.get("per_session", {}).items():
            if not info["intact"]:
                _out(f"  BROKEN: {sid}")


def generate_latex(result: ValidationResult) -> str:
    lines = []

    # Table 1: Signal comparison
    lines.append("% Table 1: Signal validation")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Distribution comparison between real keystroke data "
                 f"({result.metadata['n_checkpoints']} checkpoints from "
                 f"{result.metadata['n_sessions']} sessions) and EmbodiedScholar simulation "
                 f"({result.metadata['n_sim_traces']} traces). "
                 "KS: Kolmogorov-Smirnov statistic; $d$: Cohen's effect size; "
                 "$W_1$: Wasserstein distance.}")
    lines.append("\\label{tab:keystroke-validation}")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Signal & Real Mdn & Sim Mdn & KS & $p$ & $d$ & $W_1$ \\\\")
    lines.append("\\midrule")
    for c in result.signal_comparisons:
        name = c.signal_name.replace("_", "\\_")
        sig = "$^*$" if c.ks_pvalue < 0.05 else ""
        lines.append(
            f"{name} & {c.real_median:.3f} & {c.sim_median:.3f} & "
            f"{c.ks_statistic:.3f} & {c.ks_pvalue:.3f}{sig} & "
            f"{c.cohens_d:.3f} & {c.wasserstein_distance:.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: Phase proportions
    real_p = result.phase_comparison["real"]
    sim_p = result.phase_comparison["simulated"]
    lines.append("% Table 2: Phase proportions")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Flower \\& Hayes (1981) three-phase proportions: real vs simulated.}")
    lines.append("\\label{tab:phase-validation}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Phase & Real (\\%) & Simulated (\\%) \\\\")
    lines.append("\\midrule")
    for phase_name in ["planning", "translating", "revising"]:
        lines.append(
            f"{phase_name.capitalize()} & {real_p[phase_name]*100:.1f} & {sim_p[phase_name]*100:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate embodied simulation against real keystroke data"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--db", type=Path,
        default=Path.home() / ".local" / "data" / "keystrokes.db",
        help="SQLite database path (default: ~/.local/data/keystrokes.db)",
    )
    source.add_argument("--checkpoint-csv", type=Path, help="Pre-exported checkpoint CSV")

    parser.add_argument("-n", "--n-sim-traces", type=int, default=200)
    parser.add_argument("--n-events", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, help="JSON output path")
    parser.add_argument("--output-latex", type=Path, help="LaTeX table output path")
    parser.add_argument("--quick", action="store_true", help="Reduced simulation (n=30)")
    parser.add_argument("--app-filter", type=str, help="Filter by app bundle ID substring")
    parser.add_argument("--session", type=str, help="Specific session UUID")

    args = parser.parse_args()

    if args.quick:
        args.n_sim_traces = 30
        args.n_events = 30

    _out("=" * 60)
    _out("  KEYSTROKE VALIDATION PIPELINE")
    _out("  Real typing data vs EmbodiedScholar simulation")
    _out("=" * 60)

    db = None if args.checkpoint_csv else args.db
    result = run_validation(
        db_path=db,
        checkpoint_csv=args.checkpoint_csv,
        n_sim_traces=args.n_sim_traces,
        n_events=args.n_events,
        seed=args.seed,
        app_filter=args.app_filter,
        session_filter=args.session,
    )

    # Print results
    if result.sessions:
        print_session_table(result.sessions)
    print_signal_table(result.signal_comparisons)
    print_phase_table(result.phase_comparison)
    print_chain_integrity(result.chain_integrity)

    # Summary
    n_sig = sum(1 for c in result.signal_comparisons if c.ks_pvalue < 0.05)
    n_total = len(result.signal_comparisons)
    _out("\n" + "=" * 60)
    _out(f"  SUMMARY: {n_total - n_sig}/{n_total} signals show no significant difference (p > 0.05)")
    _out(f"  Checkpoints: {result.metadata['n_checkpoints']}  |  "
          f"Sessions: {result.metadata['n_sessions']}  |  "
          f"Sim traces: {result.metadata['n_sim_traces']}")
    if result.chain_integrity["all_intact"]:
        _out(f"  Hash chain: INTACT ({result.chain_integrity['total_links_verified']} links verified)")
    _out("=" * 60)

    # JSON output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "metadata": result.metadata,
            "sessions": [asdict(s) for s in result.sessions],
            "signal_comparisons": [asdict(c) for c in result.signal_comparisons],
            "phase_comparison": result.phase_comparison,
            "chain_integrity": result.chain_integrity,
        }
        args.output.write_text(json.dumps(out, indent=2, default=str))
        _out(f"\nJSON written to {args.output}")

    # LaTeX output
    if args.output_latex:
        args.output_latex.parent.mkdir(parents=True, exist_ok=True)
        args.output_latex.write_text(generate_latex(result))
        _out(f"LaTeX written to {args.output_latex}")


if __name__ == "__main__":
    main()
