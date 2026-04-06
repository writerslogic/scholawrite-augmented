"""Impossible Forgery Experiment — tests whether adversaries who know
the detection criteria can forge plausible causal traces.

Key insight: a sufficiently informed adversary can reproduce all structural
properties measured by consciousness-correlate signals. This proves that
process integrity verification requires observational privilege — access
to raw writing process data, not derived features.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .causal_core import (
    IrreversibleProcessEngine,
    LexicalIntention,
)
from .config import get_sim_config
from .consciousness_signatures import _WEIGHTS as CONSCIOUSNESS_WEIGHTS
from .consciousness_signatures import compute_consciousness_signatures
from .embodied import EmbodiedScholar
from .metrics import auc, compute_causal_signatures, granger_causality_test
from .schema import CausalEvent

__all__ = ["ForgedTraceGenerator", "AdversarialEvaluator", "_bootstrap_auc"]

# Adversary tier names for consistent reporting
TIER_NAIVE = "naive"
TIER_STATISTICAL = "statistical"
TIER_REVERSE_ENGINEERED = "reverse_engineered"
TIER_EXPERT = "expert"
ALL_TIERS = [TIER_NAIVE, TIER_STATISTICAL, TIER_REVERSE_ENGINEERED, TIER_EXPERT]


# ── Vocabulary for synthetic traces ──────────────────────────────────────

_ACADEMIC_WORDS = [
    "The", "methodology", "establishes", "a", "robust", "baseline",
    "for", "comparative", "analysis", "of", "emerging", "patterns",
    "in", "empirical", "research", "paradigm", "framework", "this",
    "study", "demonstrates", "that", "underlying", "assumptions",
    "remain", "foundational", "to", "subsequent", "investigations",
    "however", "nevertheless", "consequently", "furthermore",
]

_FAILURE_MODES = ["lexical_starvation", "syntactic_collapse"]
_REPAIR_WORDS = ["framework", "concept", "element", "thus,", "so,", "consequently,"]


def _deterministic_hash(seed: int, idx: int) -> int:
    return int(hashlib.md5(f"{seed}:{idx}".encode()).hexdigest(), 16)


class ForgedTraceGenerator:
    """Generates causal traces with varying degrees of forgery sophistication."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    # ── Strategy 1: Naive forgery ────────────────────────────────────

    def generate_naive_forgery(
        self,
        text: str,
        target_coupling: float = -0.3,
        target_locality: float = 2.0,
        n_events: int = 30,
    ) -> List[CausalEvent]:
        """Create a trace that matches target aggregate stats but lacks
        authentic temporal micro-structure.

        Randomly assigns failures to hit target coupling, then places repairs
        to hit target locality.
        """
        rng = random.Random(self.seed)
        words = text.split() or _ACADEMIC_WORDS[:n_events]
        if len(words) < n_events:
            words = (words * ((n_events // len(words)) + 1))[:n_events]

        # Decide how many failures: ~20-30 % of events
        n_failures = max(3, int(n_events * rng.uniform(0.2, 0.3)))
        failure_positions = sorted(rng.sample(range(n_events), n_failures))

        # Place repairs near failures to approximate target locality
        repair_positions: set[int] = set()
        target_dist = max(1, int(target_locality))
        for fp in failure_positions:
            rp = min(fp + rng.randint(1, target_dist), n_events - 1)
            repair_positions.add(rp)

        events: List[CausalEvent] = []
        glucose = 0.95
        for i in range(n_events):
            word = words[i % len(words)]
            is_failure = i in failure_positions
            is_repair = i in repair_positions

            # Naive: glucose decreases linearly (no metabolic coupling)
            glucose -= rng.uniform(0.002, 0.006)
            glucose = max(0.05, glucose)

            if is_failure:
                fm = rng.choice(_FAILURE_MODES)
                repair_word = rng.choice(_REPAIR_WORDS)
                events.append(CausalEvent(
                    intention=word,
                    actual_output=repair_word,
                    status="repair",
                    failure_mode=fm,
                    repair_artifact=repair_word,
                    glucose_at_event=round(glucose, 4),
                    latency_ms=round(rng.uniform(180, 300), 2),
                    syntactic_complexity=round(rng.uniform(4.0, 8.0), 1),
                ))
            elif is_repair:
                events.append(CausalEvent(
                    intention=word,
                    actual_output=word,
                    status="success",
                    failure_mode=None,
                    repair_artifact=word,
                    glucose_at_event=round(glucose, 4),
                    latency_ms=round(rng.uniform(100, 180), 2),
                    syntactic_complexity=round(rng.uniform(2.0, 5.0), 1),
                ))
            else:
                events.append(CausalEvent(
                    intention=word,
                    actual_output=word,
                    status="success",
                    failure_mode=None,
                    repair_artifact=None,
                    glucose_at_event=round(glucose, 4),
                    latency_ms=round(rng.uniform(100, 180), 2),
                    syntactic_complexity=round(rng.uniform(1.0, 6.0), 1),
                ))

        return events

    # ── Strategy 2: Statistical forgery ──────────────────────────────

    def generate_statistical_forgery(
        self,
        text: str,
        authentic_trace: List[CausalEvent],
    ) -> List[CausalEvent]:
        """Matches marginal distributions of the authentic trace but
        destroys temporal ordering by shuffling failure positions.
        """
        rng = random.Random(self.seed)
        n = len(authentic_trace)
        if n == 0:
            return []

        # Collect marginal distributions from authentic trace
        glucoses = [e.glucose_at_event for e in authentic_trace]
        latencies = [e.latency_ms for e in authentic_trace]
        complexities = [e.syntactic_complexity for e in authentic_trace]
        failure_indices = [i for i, e in enumerate(authentic_trace) if e.status != "success"]
        repair_indices = [i for i, e in enumerate(authentic_trace) if e.repair_artifact]

        # Shuffle failure positions (preserves count, destroys temporal structure)
        n_failures = len(failure_indices)
        new_failure_positions = set(rng.sample(range(n), min(n_failures, n)))

        # Place repairs randomly near new failures
        new_repair_positions: set[int] = set()
        for fp in new_failure_positions:
            rp = min(fp + rng.randint(1, 3), n - 1)
            new_repair_positions.add(rp)

        # Shuffle glucose values (preserves distribution, destroys monotonicity)
        shuffled_glucose = list(glucoses)
        rng.shuffle(shuffled_glucose)

        # Sort glucose to be roughly decreasing but with noise
        # (statistical forgery tries to look plausible)
        shuffled_glucose.sort(reverse=True)
        for i in range(len(shuffled_glucose)):
            shuffled_glucose[i] += rng.gauss(0, 0.01)
            shuffled_glucose[i] = max(0.05, min(1.0, shuffled_glucose[i]))

        words = text.split() or [e.intention for e in authentic_trace]
        events: List[CausalEvent] = []
        for i in range(n):
            word = words[i % len(words)] if words else f"word{i}"
            is_failure = i in new_failure_positions

            if is_failure:
                fm = rng.choice(_FAILURE_MODES)
                repair_word = rng.choice(_REPAIR_WORDS)
                events.append(CausalEvent(
                    intention=word,
                    actual_output=repair_word,
                    status="repair",
                    failure_mode=fm,
                    repair_artifact=repair_word,
                    glucose_at_event=round(shuffled_glucose[i], 4),
                    latency_ms=round(rng.choice(latencies) + rng.gauss(0, 10), 2),
                    syntactic_complexity=round(rng.choice(complexities), 1),
                ))
            elif i in new_repair_positions:
                events.append(CausalEvent(
                    intention=word,
                    actual_output=word,
                    status="success",
                    failure_mode=None,
                    repair_artifact=word,
                    glucose_at_event=round(shuffled_glucose[i], 4),
                    latency_ms=round(rng.choice(latencies) + rng.gauss(0, 10), 2),
                    syntactic_complexity=round(rng.choice(complexities), 1),
                ))
            else:
                events.append(CausalEvent(
                    intention=word,
                    actual_output=word,
                    status="success",
                    failure_mode=None,
                    repair_artifact=None,
                    glucose_at_event=round(shuffled_glucose[i], 4),
                    latency_ms=round(rng.choice(latencies) + rng.gauss(0, 10), 2),
                    syntactic_complexity=round(rng.choice(complexities), 1),
                ))

        return events

    # ── Strategy 3: Reverse-engineered forgery ───────────────────────

    def generate_reverse_engineered_forgery(
        self,
        text: str,
        author_id: str = "forged_author",
        n_events: int = 30,
    ) -> Tuple[List[CausalEvent], dict]:
        """Runs a real EmbodiedScholar simulation but resets glucose for
        each token group, destroying the irreversible depletion signature.

        Returns (forged_trace, metadata).
        """
        rng = random.Random(self.seed)
        words = text.split() or _ACADEMIC_WORDS[:n_events]
        if len(words) < n_events:
            words = (words * ((n_events // len(words)) + 1))[:n_events]

        events: List[CausalEvent] = []
        reset_interval = max(3, n_events // 5)

        for i in range(n_events):
            # Fresh glucose every reset_interval tokens (the forgery flaw)
            if i % reset_interval == 0:
                author = EmbodiedScholar(author_id, initial_glucose=1.0)
                engine = IrreversibleProcessEngine(author)

            word = words[i]
            depth = rng.uniform(1.0, 8.0)
            rarity = rng.uniform(0.1, 0.9)
            cost = rng.uniform(0.01, 0.05)

            intention = LexicalIntention(word, depth, rarity, cost)
            output = engine.execute(intention)

            evt = engine.trace[-1]
            fm = evt.failure_mode
            events.append(CausalEvent(
                intention=word,
                actual_output=output.strip() if output else word,
                status="repair" if fm else "success",
                failure_mode=fm,
                repair_artifact=output.strip() if fm else None,
                glucose_at_event=round(evt.glucose_before, 4),
                latency_ms=round(evt.latency_ms, 2),
                syntactic_complexity=round(depth, 1),
            ))

        metadata = {
            "strategy": "reverse_engineered",
            "reset_interval": reset_interval,
            "author_id": author_id,
            "n_events": n_events,
        }
        return events, metadata

    # ── Strategy 4: Expert forgery ─────────────────────────────────────

    def generate_expert_forgery(
        self,
        text: str,
        author_id: str = "expert_forger",
        n_events: int = 30,
    ) -> Tuple[List[CausalEvent], dict]:
        """Expert adversary that knows all 6 consciousness signals.

        Defeats detection via 5 techniques, each targeting specific signals:

        Technique 1: full_simulation_no_reset
          Implementation: Uses real EmbodiedScholar + IrreversibleProcessEngine
            with no glucose resets (unlike reverse_engineered tier).
          Defeats: causal_dag (preserves Granger structure between glucose→latency),
            causal_concentration (engine's natural DAG shape preserved),
            cross_channel_mi (glucose→latency coupling emerges from simulation).
          Code: lines creating `author` and `engine`, calling `engine.execute()`.

        Technique 2: glucose_gated_complexity
          Implementation: Explicit depth capping: max_depth = 3.0 + glucose * 5.0.
            At low glucose, syntactic depth is constrained to [1, ~3-5].
          Defeats: causal_dag (fatigue→complexity coupling matches human pattern),
            free_energy (creates realistic 3-phase trajectory as glucose declines).

        Technique 3: glucose_gated_failure
          Implementation: At glucose < 0.65 + high depth, forces high lexical rarity
            (0.7-0.95) to trigger lexical_starvation failures. Graded: glucose < 0.5
            uses moderate rarity (0.5-0.8).
          Defeats: causal_dag (failure distribution correlates with glucose state),
            causal_concentration (failures cluster at correct glucose levels).

        Technique 4: post_failure_adaptation
          Implementation: After any failure, caps syntactic depth for next 3 tokens
            (depth ≤ 2.0 + events_since_failure * 0.5), producing a recovery ramp.
          Defeats: adaptation (mimics post-failure complexity recovery rate seen
            in human traces).

        Technique 5: embodied_latency_coupling
          Implementation: Emergent from Technique 1 — the EmbodiedScholar naturally
            couples latency to glucose via its attention_capacity and fatigue model.
            No additional code beyond running the simulation.
          Defeats: cross_channel_mi (lagged MI between glucose and latency channels),
            decay_type (natural pause distributions from embodied timing model).

        Together these prove all 6 consciousness signals are reproducible by any
        system with knowledge of signal definitions + access to the embodied model.
        """
        rng = random.Random(self.seed)
        words = text.split() or _ACADEMIC_WORDS[:n_events]
        if len(words) < n_events:
            words = (words * ((n_events // len(words)) + 1))[:n_events]

        # Run a real simulation — no glucose resets (unlike reverse-engineered)
        author = EmbodiedScholar(author_id, initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)

        # Track post-failure state for adaptation mimicry
        recent_failure = False
        events_since_failure = 0

        events: List[CausalEvent] = []
        for i in range(n_events):
            word = words[i]

            # Expert trick 1: gate complexity on glucose level
            # Low glucose → lower syntactic depth (mimics human fatigue)
            glucose = author.glucose
            max_depth = 3.0 + glucose * 5.0  # range [3, 8] based on glucose
            depth = rng.uniform(1.0, max_depth)

            # Expert trick 2: gate failure probability on glucose+complexity
            # At low glucose with high complexity, inject failures manually
            # by choosing high rarity (triggers lexical_starvation)
            if glucose < 0.65 and depth > 4.0:
                rarity = rng.uniform(0.7, 0.95)  # high rarity forces failure
            elif glucose < 0.5:
                rarity = rng.uniform(0.5, 0.8)
            else:
                rarity = rng.uniform(0.1, 0.5)

            # Expert trick 3: post-failure adaptation
            # After a failure, reduce complexity for next few tokens
            if recent_failure and events_since_failure < 3:
                depth = min(depth, 2.0 + events_since_failure * 0.5)
                events_since_failure += 1
            elif events_since_failure >= 3:
                recent_failure = False

            cost = rng.uniform(0.01, 0.05)
            intention = LexicalIntention(word, depth, rarity, cost)
            output = engine.execute(intention)

            evt = engine.trace[-1]
            fm = evt.failure_mode
            if fm:
                recent_failure = True
                events_since_failure = 0

            events.append(CausalEvent(
                intention=word,
                actual_output=output.strip() if output else word,
                status="repair" if fm else "success",
                failure_mode=fm,
                repair_artifact=output.strip() if fm else None,
                glucose_at_event=round(evt.glucose_before, 4),
                latency_ms=round(evt.latency_ms, 2),
                syntactic_complexity=round(depth, 1),
            ))

        metadata = {
            "strategy": "expert",
            "author_id": author_id,
            "n_events": n_events,
            "techniques": [
                "full_simulation_no_reset",
                "glucose_gated_complexity",
                "glucose_gated_failure",
                "post_failure_adaptation",
                "embodied_latency_coupling",
            ],
        }
        return events, metadata


# ── Feature extraction for discrimination ────────────────────────────────

def _extract_features(trace: List[CausalEvent]) -> Dict[str, float]:
    """Extract discriminative features from a causal trace."""
    if len(trace) < 4:
        return {
            "coupling": 0.0,
            "locality": 0.0,
            "granger_asymmetry": 0.0,
            "glucose_monotonicity": 0.0,
            "latency_glucose_corr": 0.0,
        }

    sigs = compute_causal_signatures(trace)

    # Glucose monotonicity: fraction of adjacent pairs where glucose decreases
    glucoses = [e.glucose_at_event for e in trace]
    mono_count = sum(1 for i in range(len(glucoses) - 1) if glucoses[i + 1] <= glucoses[i] + 0.0001)
    glucose_mono = mono_count / max(1, len(glucoses) - 1)

    # Latency-glucose correlation (human writing: higher latency at lower glucose)
    latencies = [e.latency_ms for e in trace]
    latency_glucose_corr = 0.0
    if len(trace) > 3:
        try:
            from statistics import stdev
            mu_g, mu_l = mean(glucoses), mean(latencies)
            std_g, std_l = stdev(glucoses), stdev(latencies)
            if std_g > 0 and std_l > 0:
                latency_glucose_corr = sum(
                    (g - mu_g) * (lat - mu_l) for g, lat in zip(glucoses, latencies)
                ) / ((len(glucoses) - 1) * std_g * std_l)
        except (ValueError, ZeroDivisionError):
            pass

    granger = granger_causality_test(trace)

    return {
        "coupling": sigs.get("coupling", 0.0),
        "locality": sigs.get("locality", 0.0),
        "granger_asymmetry": granger,
        "glucose_monotonicity": round(glucose_mono, 4),
        "latency_glucose_corr": round(latency_glucose_corr, 4),
    }


def _generate_authentic_trace(seed: int, n_events: int = 30) -> List[CausalEvent]:
    """Generate a single authentic trace via IrreversibleProcessEngine."""
    rng = random.Random(seed)
    author = EmbodiedScholar(f"author_{seed}", initial_glucose=1.0)
    engine = IrreversibleProcessEngine(author)

    words = list(_ACADEMIC_WORDS)
    rng.shuffle(words)
    if len(words) < n_events:
        words = (words * ((n_events // len(words)) + 1))[:n_events]

    for i in range(n_events):
        word = words[i]
        depth = rng.uniform(1.0, 8.0)
        rarity = rng.uniform(0.1, 0.9)
        cost = rng.uniform(0.01, 0.05)
        engine.execute(LexicalIntention(word, depth, rarity, cost))

    # Convert ExecutionEvent -> CausalEvent
    events: List[CausalEvent] = []
    for evt in engine.trace:
        fm = evt.failure_mode
        events.append(CausalEvent(
            intention=evt.intention.target,
            actual_output=evt.actual_output.strip() if evt.actual_output else evt.intention.target,
            status="repair" if fm else "success",
            failure_mode=fm,
            repair_artifact=evt.actual_output.strip() if fm else None,
            glucose_at_event=round(evt.glucose_before, 4),
            latency_ms=round(evt.latency_ms, 2),
            syntactic_complexity=round(evt.intention.syntactic_depth, 1),
        ))

    return events


def _extract_consciousness_signals(trace: List[CausalEvent]) -> Dict[str, float]:
    """Extract the 6 discriminative consciousness-correlate signals from a trace.

    These are the signals that the expert adversary targets.
    """
    if len(trace) < 4:
        return {
            "causal_dag": 0.0,
            "causal_concentration": 0.0,
            "cross_channel_mi": 0.0,
            "decay_type": 0.0,
            "free_energy": 0.0,
            "adaptation": 0.0,
        }

    result = compute_consciousness_signatures(trace)
    norm = result.signal_breakdown.get("normalized", {})
    return {
        "causal_dag": norm.get("causal_dag", 0.0),
        "causal_concentration": norm.get("causal_concentration", 0.0),
        "cross_channel_mi": norm.get("cross_channel_mi", 0.0),
        "decay_type": norm.get("decay_type", 0.0),
        "free_energy": norm.get("free_energy", 0.0),
        "adaptation": norm.get("adaptation", 0.0),
    }


def _bootstrap_auc(
    y_true: List[float],
    y_score: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute AUC with bootstrap confidence interval.

    Returns (auc, ci_lower, ci_upper).
    """
    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    n = len(y_true_arr)

    # Point estimate
    point_auc = auc(list(y_true_arr), list(y_score_arr))

    # Bootstrap resampling
    rng = np.random.default_rng(seed=42)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bt = y_true_arr[idx]
        bs = y_score_arr[idx]
        # Skip degenerate samples (all same label)
        if len(np.unique(bt)) < 2:
            continue
        boot_aucs.append(auc(list(bt), list(bs)))

    if not boot_aucs:
        return (point_auc, point_auc, point_auc)

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_aucs, 100 * alpha / 2))
    upper = float(np.percentile(boot_aucs, 100 * (1.0 - alpha / 2)))
    return (round(point_auc, 4), round(lower, 4), round(upper, 4))


class AdversarialEvaluator:
    """Evaluates whether forged traces can be distinguished from authentic ones.

    Produces a signal × tier AUC matrix showing per-signal discrimination
    power against each adversary tier. This is the primary reporting format
    for the impossibility boundary analysis.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def evaluate_forgery_detection(
        self,
        authentic_traces: List[List[CausalEvent]],
        forged_traces: List[List[CausalEvent]],
    ) -> dict:
        """Compute discrimination metrics between authentic and forged traces.

        Returns per-feature AUC and combined AUC (legacy format for compat).
        """
        auth_features = [_extract_features(t) for t in authentic_traces]
        forge_features = [_extract_features(t) for t in forged_traces]

        if not auth_features or not forge_features:
            return {"per_feature_auc": {}, "combined_auc": 0.0}

        feature_names = list(auth_features[0].keys())
        per_feature_auc: Dict[str, float] = {}

        y_true = [1.0] * len(auth_features) + [0.0] * len(forge_features)

        for feat in feature_names:
            y_score = [f[feat] for f in auth_features] + [f[feat] for f in forge_features]
            feat_auc = auc(y_true, y_score)
            per_feature_auc[feat] = round(max(feat_auc, 1.0 - feat_auc), 4)

        combined_scores: List[float] = []
        for features in auth_features + forge_features:
            score = (
                0.3 * features["glucose_monotonicity"]
                + 0.25 * features["granger_asymmetry"]
                + 0.2 * abs(features["coupling"])
                + 0.15 * abs(features["latency_glucose_corr"])
                + 0.1 * features["locality"]
            )
            combined_scores.append(score)

        combined_auc_val = auc(y_true, combined_scores)
        combined_auc_val = round(combined_auc_val, 4)

        return {
            "per_feature_auc": per_feature_auc,
            "combined_auc": combined_auc_val,
            "n_authentic": len(authentic_traces),
            "n_forged": len(forged_traces),
        }

    def evaluate_consciousness_discrimination(
        self,
        authentic_traces: List[List[CausalEvent]],
        forged_traces: List[List[CausalEvent]],
    ) -> Dict[str, Any]:
        """Evaluate per-signal AUC using the 6 consciousness-correlate signals.

        Returns per-signal AUC, composite AUC, and raw score distributions.
        """
        auth_signals = [_extract_consciousness_signals(t) for t in authentic_traces]
        forge_signals = [_extract_consciousness_signals(t) for t in forged_traces]

        if not auth_signals or not forge_signals:
            return {"per_signal_auc": {}, "composite_auc": 0.0}

        signal_names = list(auth_signals[0].keys())
        y_true = [1.0] * len(auth_signals) + [0.0] * len(forge_signals)
        all_signals = auth_signals + forge_signals

        per_signal_auc: Dict[str, float] = {}
        for sig in signal_names:
            scores = [s[sig] for s in all_signals]
            sig_auc = auc(y_true, scores)
            per_signal_auc[sig] = round(sig_auc, 4)

        # Composite: weighted by discrimination power (matches consciousness_signatures._WEIGHTS)
        composite_scores = [
            sum(CONSCIOUSNESS_WEIGHTS.get(k, 0.0) * v for k, v in s.items())
            for s in all_signals
        ]
        composite_auc = round(auc(y_true, composite_scores), 4)

        # Distribution stats
        auth_composites = composite_scores[:len(auth_signals)]
        forge_composites = composite_scores[len(auth_signals):]

        return {
            "per_signal_auc": per_signal_auc,
            "composite_auc": composite_auc,
            "n_authentic": len(authentic_traces),
            "n_forged": len(forged_traces),
            "authentic_mean": round(mean(auth_composites), 4) if auth_composites else 0.0,
            "forged_mean": round(mean(forge_composites), 4) if forge_composites else 0.0,
        }

    def _generate_tier_traces(
        self,
        n_samples: int,
        n_events: int,
        authentic_traces: List[List[CausalEvent]],
    ) -> Dict[str, List[List[CausalEvent]]]:
        """Generate forged traces for all adversary tiers."""
        sample_text = " ".join(_ACADEMIC_WORDS * 3)
        traces_by_tier: Dict[str, List[List[CausalEvent]]] = {}

        # Naive
        naive: List[List[CausalEvent]] = []
        for i in range(n_samples):
            gen = ForgedTraceGenerator(seed=self.seed + 1000 + i)
            naive.append(gen.generate_naive_forgery(sample_text, n_events=n_events))
        traces_by_tier[TIER_NAIVE] = naive

        # Statistical
        stat: List[List[CausalEvent]] = []
        for i in range(n_samples):
            gen = ForgedTraceGenerator(seed=self.seed + 2000 + i)
            template = authentic_traces[i % len(authentic_traces)]
            stat.append(gen.generate_statistical_forgery(sample_text, template))
        traces_by_tier[TIER_STATISTICAL] = stat

        # Reverse-engineered
        rev: List[List[CausalEvent]] = []
        for i in range(n_samples):
            gen = ForgedTraceGenerator(seed=self.seed + 3000 + i)
            trace, _ = gen.generate_reverse_engineered_forgery(
                sample_text, author_id=f"forger_{i}", n_events=n_events,
            )
            rev.append(trace)
        traces_by_tier[TIER_REVERSE_ENGINEERED] = rev

        # Expert
        expert: List[List[CausalEvent]] = []
        for i in range(n_samples):
            gen = ForgedTraceGenerator(seed=self.seed + 4000 + i)
            trace, _ = gen.generate_expert_forgery(
                sample_text, author_id=f"expert_{i}", n_events=n_events,
            )
            expert.append(trace)
        traces_by_tier[TIER_EXPERT] = expert

        return traces_by_tier

    def run_signal_ablation(
        self,
        n_samples: int = 50,
        n_events: int = 30,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Full ablation study over the 6 consciousness-correlate signals.

        For each adversary tier, computes:
        - Full composite AUC (using actual _WEIGHTS, with bootstrap CI)
        - Leave-one-out AUC (signal removed, remaining weights renormalized)
        - Standalone AUC (single signal only, weight=1.0)
        - Delta AUC (full minus leave-one-out)
        - Pairwise signal correlation matrix across all traces
        """
        from .metrics import bootstrap_auc_ci

        signal_names = list(CONSCIOUSNESS_WEIGHTS.keys())

        # Generate authentic traces
        authentic_traces: List[List[CausalEvent]] = []
        for i in range(n_samples):
            authentic_traces.append(
                _generate_authentic_trace(seed=self.seed + i, n_events=n_events)
            )

        # Generate forged traces for all tiers
        traces_by_tier = self._generate_tier_traces(
            n_samples, n_events, authentic_traces,
        )

        # Extract normalized signals once per trace (reused across ablations)
        auth_signals = [_extract_consciousness_signals(t) for t in authentic_traces]

        def _weighted_composite(
            signals: List[Dict[str, float]], weights: Dict[str, float],
        ) -> List[float]:
            """Compute weighted composite score for each trace."""
            return [
                sum(weights.get(k, 0.0) * s.get(k, 0.0) for k in signal_names)
                for s in signals
            ]

        def _renormalized_weights(exclude: str) -> Dict[str, float]:
            """Return weights with `exclude` zeroed and rest renormalized to sum=1."""
            remaining = {k: v for k, v in CONSCIOUSNESS_WEIGHTS.items() if k != exclude}
            total = sum(remaining.values())
            if total < 1e-10:
                return {k: 0.0 for k in CONSCIOUSNESS_WEIGHTS}
            normed = {k: v / total for k, v in remaining.items()}
            normed[exclude] = 0.0
            return normed

        # Per-tier ablation results
        tier_results: Dict[str, Any] = {}

        for tier in ALL_TIERS:
            forged = traces_by_tier[tier]
            forge_signals = [_extract_consciousness_signals(t) for t in forged]
            all_signals = auth_signals + forge_signals
            y_true = [1.0] * len(auth_signals) + [0.0] * len(forge_signals)

            # Full composite AUC (actual weights)
            full_scores = _weighted_composite(all_signals, CONSCIOUSNESS_WEIGHTS)
            full_auc, full_lo, full_hi = bootstrap_auc_ci(
                y_true, full_scores, n_bootstrap=n_bootstrap, ci_level=ci_level,
            )

            # Leave-one-out and standalone for each signal
            leave_one_out: Dict[str, Dict[str, Any]] = {}
            standalone: Dict[str, Dict[str, Any]] = {}

            for sig in signal_names:
                # Leave-one-out: renormalize remaining weights
                loo_weights = _renormalized_weights(sig)
                loo_scores = _weighted_composite(all_signals, loo_weights)
                loo_auc, loo_lo, loo_hi = bootstrap_auc_ci(
                    y_true, loo_scores, n_bootstrap=n_bootstrap, ci_level=ci_level,
                )
                delta = full_auc - loo_auc
                leave_one_out[sig] = {
                    "auc": loo_auc,
                    "ci_lower": loo_lo,
                    "ci_upper": loo_hi,
                    "delta": round(delta, 4),
                }

                # Standalone: only this signal (weight=1.0)
                solo_weights = {k: (1.0 if k == sig else 0.0) for k in signal_names}
                solo_scores = _weighted_composite(all_signals, solo_weights)
                solo_auc, solo_lo, solo_hi = bootstrap_auc_ci(
                    y_true, solo_scores, n_bootstrap=n_bootstrap, ci_level=ci_level,
                )
                standalone[sig] = {
                    "auc": solo_auc,
                    "ci_lower": solo_lo,
                    "ci_upper": solo_hi,
                }

            tier_results[tier] = {
                "full_composite": {
                    "auc": full_auc, "ci_lower": full_lo, "ci_upper": full_hi,
                },
                "leave_one_out": leave_one_out,
                "standalone": standalone,
            }

        # Signal correlation matrix (computed over all traces: authentic + all tiers)
        all_trace_signals = list(auth_signals)
        for tier in ALL_TIERS:
            forged = traces_by_tier[tier]
            all_trace_signals.extend(
                _extract_consciousness_signals(t) for t in forged
            )
        correlation_matrix = self._compute_signal_correlation(
            all_trace_signals, signal_names,
        )

        return {
            "weights": dict(CONSCIOUSNESS_WEIGHTS),
            "signal_names": signal_names,
            "tier_results": tier_results,
            "correlation_matrix": correlation_matrix,
            "n_samples": n_samples,
            "n_events": n_events,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
        }

    @staticmethod
    def _compute_signal_correlation(
        signals: List[Dict[str, float]],
        signal_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Pairwise Pearson correlation matrix across signal values."""
        n = len(signals)
        if n < 3:
            return {a: {b: 0.0 for b in signal_names} for a in signal_names}

        # Extract columns
        cols: Dict[str, List[float]] = {
            sig: [s[sig] for s in signals] for sig in signal_names
        }
        # Precompute means and stds
        stats: Dict[str, Tuple[float, float]] = {}
        for sig in signal_names:
            vals = cols[sig]
            mu = sum(vals) / n
            var = sum((v - mu) ** 2 for v in vals) / (n - 1)
            stats[sig] = (mu, var ** 0.5)

        matrix: Dict[str, Dict[str, float]] = {}
        for a in signal_names:
            matrix[a] = {}
            mu_a, std_a = stats[a]
            for b in signal_names:
                if a == b:
                    matrix[a][b] = 1.0
                    continue
                mu_b, std_b = stats[b]
                if std_a < 1e-10 or std_b < 1e-10:
                    matrix[a][b] = 0.0
                    continue
                cov = sum(
                    (cols[a][i] - mu_a) * (cols[b][i] - mu_b)
                    for i in range(n)
                ) / (n - 1)
                matrix[a][b] = round(cov / (std_a * std_b), 4)
        return matrix

    def run_full_evaluation(
        self,
        n_samples: int = 50,
        n_events: int = 30,
        include_consciousness: bool = True,
    ) -> dict:
        """Generate authentic and forged traces, report detection rates per strategy.

        When include_consciousness=True (default), also produces the signal × tier
        AUC matrix using the 6 consciousness-correlate signals.
        """
        # Generate authentic traces
        authentic_traces: List[List[CausalEvent]] = []
        for i in range(n_samples):
            trace = _generate_authentic_trace(seed=self.seed + i, n_events=n_events)
            authentic_traces.append(trace)

        # Generate all adversary tiers
        traces_by_tier = self._generate_tier_traces(n_samples, n_events, authentic_traces)

        # Legacy evaluation (5 core features)
        legacy_results: Dict[str, Any] = {}
        for tier, forged in traces_by_tier.items():
            legacy_results[tier] = self.evaluate_forgery_detection(authentic_traces, forged)

        result: Dict[str, Any] = {
            "n_samples": n_samples,
            "n_events": n_events,
            # Legacy format for backward compat
            "naive_forgery": legacy_results[TIER_NAIVE],
            "statistical_forgery": legacy_results[TIER_STATISTICAL],
            "reverse_engineered_forgery": legacy_results[TIER_REVERSE_ENGINEERED],
            "expert_forgery": legacy_results[TIER_EXPERT],
        }

        if include_consciousness:
            # Consciousness signal × tier AUC matrix
            consciousness_by_tier: Dict[str, Any] = {}
            for tier, forged in traces_by_tier.items():
                consciousness_by_tier[tier] = self.evaluate_consciousness_discrimination(
                    authentic_traces, forged,
                )

            # Build signal × tier matrix (the key output for the paper)
            signal_names = ["causal_dag", "causal_concentration", "cross_channel_mi",
                            "decay_type", "free_energy", "adaptation"]
            signal_tier_matrix: Dict[str, Dict[str, float]] = {}
            for sig in signal_names:
                signal_tier_matrix[sig] = {}
                for tier in ALL_TIERS:
                    tier_data = consciousness_by_tier.get(tier, {})
                    per_sig = tier_data.get("per_signal_auc", {})
                    signal_tier_matrix[sig][tier] = per_sig.get(sig, 0.0)

            # Composite AUC per tier
            composite_by_tier = {
                tier: consciousness_by_tier[tier].get("composite_auc", 0.0)
                for tier in ALL_TIERS
            }

            result["consciousness"] = {
                "signal_tier_matrix": signal_tier_matrix,
                "composite_by_tier": composite_by_tier,
                "per_tier_detail": consciousness_by_tier,
            }

        return result

    def profile_adversarial_cost(
        self,
        n_samples: int = 50,
        n_events: int = 30,
    ) -> Dict[str, Any]:
        """Profile wall-clock cost per adversary strategy.

        Returns timing per trace generation and total, enabling
        cost comparison across adversary tiers.
        """
        sample_text = " ".join(_ACADEMIC_WORDS * 5)
        cost_results: Dict[str, Any] = {}

        # Authentic
        t0 = time.perf_counter()
        for i in range(n_samples):
            _generate_authentic_trace(seed=self.seed + i, n_events=n_events)
        auth_time = time.perf_counter() - t0
        cost_results["authentic"] = {
            "total_ms": round(auth_time * 1000, 1),
            "per_trace_ms": round(auth_time * 1000 / n_samples, 2),
        }

        # Generate one authentic template for statistical tier
        auth_template = _generate_authentic_trace(seed=self.seed, n_events=n_events)

        tier_methods = {
            "naive": lambda i: ForgedTraceGenerator(seed=self.seed + 1000 + i).generate_naive_forgery(
                sample_text, n_events=n_events,
            ),
            "statistical": lambda i: ForgedTraceGenerator(seed=self.seed + 2000 + i).generate_statistical_forgery(
                sample_text, auth_template,
            ),
            "reverse_engineered": lambda i: ForgedTraceGenerator(seed=self.seed + 3000 + i).generate_reverse_engineered_forgery(
                sample_text, author_id=f"f_{i}", n_events=n_events,
            ),
            "expert": lambda i: ForgedTraceGenerator(seed=self.seed + 4000 + i).generate_expert_forgery(
                sample_text, author_id=f"e_{i}", n_events=n_events,
            ),
        }

        for tier, gen_fn in tier_methods.items():
            t0 = time.perf_counter()
            for i in range(n_samples):
                gen_fn(i)
            elapsed = time.perf_counter() - t0
            cost_results[tier] = {
                "total_ms": round(elapsed * 1000, 1),
                "per_trace_ms": round(elapsed * 1000 / n_samples, 2),
                "cost_ratio": round(elapsed / auth_time, 2) if auth_time > 0 else 0.0,
            }

        return cost_results

    def run_held_out_evaluation(
        self,
        n_samples: int = 100,
        n_events_list: Optional[List[int]] = None,
        seed_offset: int = 500,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        include_consciousness: bool = True,
    ) -> dict:
        """Evaluate forgery detection with bootstrap CIs across trace lengths.

        Uses held-out seeds (starting at seed_offset) that don't overlap with
        tuning seeds 0-499. Computes AUC with bootstrap 95% CI for each
        (tier, trace_length) pair.

        Returns nested dict: {n_events: {tier: {auc, ci_lower, ci_upper, n_samples}}}
        """
        if n_events_list is None:
            n_events_list = [20, 40, 60, 80]

        result: Dict[int, Dict[str, Any]] = {}

        for n_events in n_events_list:
            # Generate authentic traces with held-out seeds
            authentic_traces: List[List[CausalEvent]] = []
            for i in range(n_samples):
                trace = _generate_authentic_trace(
                    seed=seed_offset + i, n_events=n_events,
                )
                authentic_traces.append(trace)

            # Generate forged traces for all tiers
            traces_by_tier = self._generate_tier_traces(
                n_samples, n_events, authentic_traces,
            )

            tier_results: Dict[str, Any] = {}
            for tier in ALL_TIERS:
                forged = traces_by_tier[tier]

                # Extract features and build scores
                auth_features = [_extract_features(t) for t in authentic_traces]
                forge_features = [_extract_features(t) for t in forged]

                # Combined score (same weighting as evaluate_forgery_detection)
                y_true = [1.0] * len(auth_features) + [0.0] * len(forge_features)
                y_score: List[float] = []
                for features in auth_features + forge_features:
                    score = (
                        0.3 * features["glucose_monotonicity"]
                        + 0.25 * features["granger_asymmetry"]
                        + 0.2 * abs(features["coupling"])
                        + 0.15 * abs(features["latency_glucose_corr"])
                        + 0.1 * features["locality"]
                    )
                    y_score.append(score)

                auc_val, ci_lower, ci_upper = _bootstrap_auc(
                    y_true, y_score,
                    n_bootstrap=n_bootstrap, ci_level=ci_level,
                )

                tier_entry: Dict[str, Any] = {
                    "auc": auc_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_samples": n_samples,
                }

                if include_consciousness:
                    # Consciousness signal × tier with CIs
                    auth_signals = [_extract_consciousness_signals(t) for t in authentic_traces]
                    forge_signals = [_extract_consciousness_signals(t) for t in forged]
                    all_signals = auth_signals + forge_signals

                    signal_names = list(auth_signals[0].keys())
                    signal_cis: Dict[str, Dict[str, float]] = {}
                    for sig in signal_names:
                        scores = [s[sig] for s in all_signals]
                        s_auc, s_lo, s_hi = _bootstrap_auc(
                            y_true, scores,
                            n_bootstrap=n_bootstrap, ci_level=ci_level,
                        )
                        signal_cis[sig] = {
                            "auc": s_auc, "ci_lower": s_lo, "ci_upper": s_hi,
                        }

                    tier_entry["consciousness_signals"] = signal_cis

                tier_results[tier] = tier_entry

            result[n_events] = tier_results

        return result
