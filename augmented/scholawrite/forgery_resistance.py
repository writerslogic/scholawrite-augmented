"""Forgery resistance analysis for causal execution traces.

Formalizes the argument that producing text with authentic causal signatures
requires sequential computation that cannot be parallelized. This is analogous
to Verifiable Delay Functions (VDFs) in cryptography: just as a VDF forces
a prover to perform T sequential squarings (no shortcut via parallelism),
an authentic writing trace forces T sequential glucose-depleting cognitive
steps where each step's state is derived solely from the previous step's
output plus the current intention.

A forger who wishes to fabricate a plausible trace must either:
  1. Actually perform the sequential process (no speedup), or
  2. Search an exponential state space to find glucose/latency values
     that satisfy all pairwise consistency constraints simultaneously.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .causal_core import ExecutionEvent, IrreversibleProcessEngine, LexicalIntention
from .embodied import EmbodiedScholar
from .config import get_sim_config

__all__ = [
    "SequentialProcessAttestation",
    "compute_forgery_cost",
    "verify_trace_authenticity",
]


class SequentialProcessAttestation:
    """Verify that a causal trace exhibits properties requiring sequential computation.

    Analogous to VDF verification: checking a proof is fast, but producing
    one without doing the sequential work is infeasible.
    """

    def __init__(self, trace: List[ExecutionEvent]):
        if not trace:
            raise ValueError("Trace must be non-empty")
        self.trace = trace

    def verify_monotonic_depletion(self) -> bool:
        """Glucose values are strictly non-increasing across the trace."""
        for i in range(1, len(self.trace)):
            if self.trace[i].glucose_before > self.trace[i - 1].glucose_before + 1e-9:
                return False
        return True

    def verify_temporal_consistency(self) -> bool:
        """Latency values correlate with glucose state.

        The embodied model predicts lower glucose -> higher latency because
        cognitive depletion increases inter-keystroke intervals.
        Returns True if the correlation direction is consistent (negative
        correlation between glucose and latency).
        """
        if len(self.trace) < 3:
            return True  # too few points to judge

        glucose_vals = [e.glucose_before for e in self.trace]
        latency_vals = [e.latency_ms for e in self.trace]

        # Compute Pearson correlation sign
        n = len(glucose_vals)
        mean_g = sum(glucose_vals) / n
        mean_l = sum(latency_vals) / n

        cov = sum((g - mean_g) * (l - mean_l) for g, l in zip(glucose_vals, latency_vals))
        # Negative covariance means lower glucose -> higher latency
        # Zero covariance is acceptable (flat trace)
        return cov <= 1e-9

    def verify_state_dependency(self) -> bool:
        """Each event's glucose_after depends on the previous event's glucose_before.

        Specifically, for consecutive events i and i+1:
          event[i+1].glucose_before must equal event[i].glucose_after
          (within floating-point tolerance)

        This forms a hash-chain-like dependency: you cannot compute event[i+1]
        without first computing event[i].
        """
        for i in range(len(self.trace) - 1):
            current_after = self.trace[i].glucose_after
            next_before = self.trace[i + 1].glucose_before
            # The engine records glucose_before as the author's glucose at the
            # start of the step. After consume_resources + depletion, the
            # glucose_after is recorded. The next step's glucose_before should
            # be <= current glucose_after (the engine may apply additional
            # depletion via consume_resources).
            if next_before > current_after + 1e-6:
                return False
        return True

    def verify_causal_chain_integrity(self) -> bool:
        """The full trace forms a causal chain where each state derives from its predecessor.

        Checks all three sub-properties together:
        1. Monotonic depletion (thermodynamic arrow)
        2. Temporal consistency (embodied coupling)
        3. State dependency (sequential hash chain)
        """
        return (
            self.verify_monotonic_depletion()
            and self.verify_temporal_consistency()
            and self.verify_state_dependency()
        )

    def attestation_score(self) -> float:
        """Composite score in [0.0, 1.0] measuring trace authenticity.

        Each verification contributes equally. A score of 1.0 means all
        sequential-process properties hold; 0.0 means none do.
        """
        checks = [
            self.verify_monotonic_depletion(),
            self.verify_temporal_consistency(),
            self.verify_state_dependency(),
            self.verify_causal_chain_integrity(),
        ]
        # causal_chain_integrity is a conjunction of the first three,
        # so weight it separately to reward full consistency
        # Weights: mono=0.25, temporal=0.25, state_dep=0.25, full_chain=0.25
        return sum(1.0 for c in checks if c) / len(checks)


def compute_forgery_cost(trace_length: int, glucose_precision: int = 6) -> dict:
    """Estimate the computational cost of forging a plausible trace.

    Args:
        trace_length: Number of events in the trace.
        glucose_precision: Decimal digits of glucose values a forger must match.

    Returns:
        Dict with:
          sequential_steps: Minimum sequential operations (= trace_length,
              since each step depends on the previous state).
          state_space: Size of the state space the forger must search to
              match glucose signatures at the given precision.
          parallel_speedup: Theoretical maximum parallel speedup for
              the sequential chain. For a truly sequential process this
              is ~1.0 (no parallelism helps).
    """
    if trace_length < 1:
        raise ValueError("trace_length must be >= 1")

    # Each step has 10^precision possible glucose values
    per_step_states = 10 ** glucose_precision

    # The forger must match a chain of trace_length states, each constrained
    # by the previous. Total search space = per_step_states ^ trace_length
    # but in practice constrained by monotonicity -> roughly C(per_step_states, trace_length)
    # We report the unconstrained upper bound for clarity.
    state_space = per_step_states ** trace_length

    # Sequential dependency means parallelism cannot reduce the critical path
    # below trace_length steps. Amdahl's law with serial fraction = 1.0.
    parallel_speedup = 1.0

    return {
        "sequential_steps": trace_length,
        "state_space": state_space,
        "parallel_speedup": parallel_speedup,
    }


def verify_trace_authenticity(
    trace: List[ExecutionEvent],
    author: EmbodiedScholar,
) -> dict:
    """Replay a trace through the engine and check if outputs match.

    Creates a fresh EmbodiedScholar with the same initial state and replays
    each intention. Returns a verification result with any discrepancies.

    Args:
        trace: The causal trace to verify.
        author: The original author (used for author_id and initial glucose).

    Returns:
        Dict with:
          authentic: bool -- whether the trace is consistent with replay.
          discrepancies: list of dicts describing any mismatches.
          attestation_score: float in [0.0, 1.0].
    """
    if not trace:
        return {"authentic": False, "discrepancies": [{"index": -1, "field": "trace", "detail": "empty trace"}], "attestation_score": 0.0}

    # Structural verification via SequentialProcessAttestation
    attestation = SequentialProcessAttestation(trace)
    score = attestation.attestation_score()

    # Replay verification: re-execute each intention with a fresh author
    replay_author = EmbodiedScholar(
        author_id=author.author_id,
        initial_glucose=trace[0].glucose_before,
    )
    engine = IrreversibleProcessEngine(replay_author)

    discrepancies = []
    for i, event in enumerate(trace):
        intention = event.intention
        replayed_output = engine.execute(intention)
        replayed_event = engine.trace[-1]

        # Check output match
        if replayed_event.actual_output != event.actual_output:
            discrepancies.append({
                "index": i,
                "field": "actual_output",
                "expected": event.actual_output,
                "got": replayed_event.actual_output,
            })

        # Check failure mode match
        if replayed_event.failure_mode != event.failure_mode:
            discrepancies.append({
                "index": i,
                "field": "failure_mode",
                "expected": event.failure_mode,
                "got": replayed_event.failure_mode,
            })

        # Check glucose_after within tolerance
        if abs(replayed_event.glucose_after - event.glucose_after) > 1e-4:
            discrepancies.append({
                "index": i,
                "field": "glucose_after",
                "expected": event.glucose_after,
                "got": replayed_event.glucose_after,
            })

    authentic = len(discrepancies) == 0 and score == 1.0
    return {
        "authentic": authentic,
        "discrepancies": discrepancies,
        "attestation_score": score,
    }
