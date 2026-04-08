#!/usr/bin/env python
"""Sensitivity analysis across simulation thresholds.

Sweeps key parameters and reports impact on causal signature distributions.
Usage: uv run python scripts/ablation_sweep.py --data-dir output/
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scholawrite.config import SimulationConfig, get_sim_config
from scholawrite.causal_core import IrreversibleProcessEngine, LexicalIntention
from scholawrite.embodied import EmbodiedScholar


SWEEP_RANGES = {
    "glucose_lexical_starvation": [0.45, 0.55, 0.65, 0.75, 0.85],
    "syntactic_collapse_base": [3.0, 3.5, 4.0, 4.5, 5.0],
    "coupling_strong_threshold": [0.4, 0.5, 0.6, 0.7, 0.8],
    "locality_human_min": [0.5, 1.0, 1.5, 2.0],
    "locality_human_max": [2.5, 3.0, 3.5, 4.0, 4.5],
    "failure_repair_cost_multiplier": [1.5, 2.0, 2.7, 3.5, 4.0],
}


def _make_test_intentions(n: int = 50) -> list[LexicalIntention]:
    """Generate a reproducible set of test intentions."""
    import random
    rng = random.Random(42)
    intentions = []
    for i in range(n):
        intentions.append(LexicalIntention(
            target=f"token_{i}",
            syntactic_depth=rng.uniform(1.0, 8.0),
            lexical_rarity=rng.uniform(0.0, 1.0),
            cognitive_cost=rng.uniform(0.01, 0.08),
        ))
    return intentions


def run_single_config(config: SimulationConfig) -> dict:
    """Run the engine with a config and return signature stats."""
    import scholawrite.config as cfg_mod
    cfg_mod.get_sim_config.cache_clear()
    original = cfg_mod.get_sim_config

    def patched():
        return config
    cfg_mod.get_sim_config = patched

    try:
        author = EmbodiedScholar("test_author", initial_glucose=config.initial_glucose)
        engine = IrreversibleProcessEngine(author=author)
        intentions = _make_test_intentions()

        for intent in intentions:
            engine.execute(intent)

        sigs = engine.compute_causal_signatures()
        return {
            "repair_locality": sigs["repair_locality"],
            "resource_coupling": sigs["resource_coupling"],
            "is_plausible": sigs["is_plausible"],
            "plausibility_score": sigs.get("plausibility_score", 0.0),
            "final_glucose": round(author.glucose, 4),
            "trace_length": len(engine.trace),
            "failure_count": sum(1 for e in engine.trace if e.failure_mode),
        }
    finally:
        cfg_mod.get_sim_config = original
        cfg_mod.get_sim_config.cache_clear()


def run_signal_subset_ablation(n_samples: int = 50, n_events: int = 30, seed: int = 42) -> dict:
    """Ablate consciousness signal subsets and report composite AUC per subset.

    Tests all leave-one-out subsets and all individual signals.
    """
    from scholawrite.adversarial import (
        AdversarialEvaluator, ForgedTraceGenerator,
        _extract_consciousness_signals, _generate_authentic_trace, _ACADEMIC_WORDS,
    )
    from scholawrite.metrics import auc, bootstrap_auc_ci

    evaluator = AdversarialEvaluator(seed=seed)

    # Generate authentic + expert-forged traces
    authentic = [_generate_authentic_trace(seed=seed + i, n_events=n_events) for i in range(n_samples)]
    sample_text = " ".join(_ACADEMIC_WORDS * 3)
    forged = []
    for i in range(n_samples):
        gen = ForgedTraceGenerator(seed=seed + 4000 + i)
        trace, _ = gen.generate_expert_forgery(sample_text, author_id=f"expert_{i}", n_events=n_events)
        forged.append(trace)

    auth_sigs = [_extract_consciousness_signals(t) for t in authentic]
    forge_sigs = [_extract_consciousness_signals(t) for t in forged]
    all_sigs = auth_sigs + forge_sigs
    y_true = [1.0] * n_samples + [0.0] * n_samples

    signal_names = list(auth_sigs[0].keys())

    # Full composite baseline
    baseline_scores = [sum(s.values()) / len(s) for s in all_sigs]
    bl_pt, bl_lo, bl_hi = bootstrap_auc_ci(y_true, baseline_scores)

    results = {"baseline": {"auc": bl_pt, "ci": [bl_lo, bl_hi]}}

    # Leave-one-out
    for sig in signal_names:
        remaining = [n for n in signal_names if n != sig]
        scores = [sum(s[r] for r in remaining) / len(remaining) for s in all_sigs]
        pt, lo, hi = bootstrap_auc_ci(y_true, scores)
        results[f"without_{sig}"] = {"auc": pt, "ci": [lo, hi], "delta": round(bl_pt - pt, 4)}

    # Standalone
    for sig in signal_names:
        scores = [s[sig] for s in all_sigs]
        pt, lo, hi = bootstrap_auc_ci(y_true, scores)
        results[f"only_{sig}"] = {"auc": pt, "ci": [lo, hi]}

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation sweep over simulation thresholds")
    parser.add_argument("--params", nargs="*", help="Parameters to sweep (default: all)")
    parser.add_argument("--signal-ablation", action="store_true",
                        help="Run consciousness signal subset ablation (leave-one-out + standalone)")
    parser.add_argument("-n", "--n-samples", type=int, default=50, help="Samples for signal ablation")
    parser.add_argument("--output", type=Path, default=Path("ablation_results.json"))
    args = parser.parse_args()

    results = {}

    if args.signal_ablation:
        print("Running consciousness signal subset ablation...")
        sig_results = run_signal_subset_ablation(n_samples=args.n_samples)
        results["signal_ablation"] = sig_results
        bl = sig_results["baseline"]
        print(f"\n  Baseline AUC: {bl['auc']:.3f} ({bl['ci'][0]:.3f}-{bl['ci'][1]:.3f})")
        for key, val in sig_results.items():
            if key.startswith("without_"):
                sig = key[len("without_"):]
                print(f"  Without {sig:<25s}: AUC={val['auc']:.3f} delta={val['delta']:+.4f}")
        for key, val in sig_results.items():
            if key.startswith("only_"):
                sig = key[len("only_"):]
                print(f"  Only {sig:<28s}: AUC={val['auc']:.3f}")

    params_to_sweep = args.params or list(SWEEP_RANGES.keys())
    base_config = get_sim_config()


    for param_name in params_to_sweep:
        if param_name not in SWEEP_RANGES:
            print(f"Unknown parameter: {param_name}, skipping")
            continue

        print(f"\nSweeping {param_name}:")
        param_results = []
        for value in SWEEP_RANGES[param_name]:
            # Create config with overridden parameter
            overrides = {param_name: value}
            config = SimulationConfig(**{
                **{f.name: getattr(base_config, f.name) for f in base_config.__dataclass_fields__.values()},
                **overrides,
            })

            sigs = run_single_config(config)
            sigs["param_value"] = value
            param_results.append(sigs)
            print(f"  {param_name}={value:.2f} -> locality={sigs['repair_locality']:.2f}, "
                  f"coupling={sigs['resource_coupling']:.3f}, "
                  f"plausible={sigs['is_plausible']}, "
                  f"failures={sigs['failure_count']}")

        results[param_name] = param_results

    # Save results
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
