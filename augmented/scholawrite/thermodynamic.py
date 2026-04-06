"""Thermodynamic irreversibility metrics for causal traces.

Measures entropy production rate — the KL divergence between forward-time
and reverse-time transition statistics. Human writing is dissipative
(positive entropy production); LLM text is near-reversible (sigma ~ 0).
"""
from __future__ import annotations

import math
from typing import List, Dict

from .schema import CausalEvent

__all__ = [
    "compute_entropy_production_rate",
    "compute_joint_entropy_production",
    "compute_time_asymmetry",
    "compute_dissipation_trajectory",
]

# Smoothing constant to avoid log(0)
_EPSILON = 1e-10
# Default number of bins for discretization
_N_BINS = 10


def _discretize(values: List[float], n_bins: int = _N_BINS) -> List[int]:
    """Map continuous values to discrete bin indices."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < _EPSILON:
        return [0] * len(values)
    return [min(n_bins - 1, int((v - lo) / (hi - lo) * n_bins)) for v in values]


def _transition_probs(bins: List[int], n_bins: int = _N_BINS) -> List[List[float]]:
    """Compute transition probability matrix P[i][j] = P(next=j | current=i)."""
    counts = [[0] * n_bins for _ in range(n_bins)]
    for t in range(len(bins) - 1):
        counts[bins[t]][bins[t + 1]] += 1

    probs = [[0.0] * n_bins for _ in range(n_bins)]
    for i in range(n_bins):
        row_sum = sum(counts[i])
        if row_sum > 0:
            for j in range(n_bins):
                probs[i][j] = (counts[i][j] + _EPSILON) / (row_sum + n_bins * _EPSILON)
        else:
            # Uniform prior for unvisited states
            for j in range(n_bins):
                probs[i][j] = 1.0 / n_bins
    return probs


def _kl_divergence_matrices(
    p_fwd: List[List[float]],
    p_rev: List[List[float]],
    state_freq: List[float],
    n_bins: int = _N_BINS,
) -> float:
    """Weighted KL divergence D_KL(P_forward || P_reverse) over state distribution."""
    kl = 0.0
    for i in range(n_bins):
        if state_freq[i] < _EPSILON:
            continue
        for j in range(n_bins):
            if p_fwd[i][j] > _EPSILON:
                kl += state_freq[i] * p_fwd[i][j] * math.log(p_fwd[i][j] / p_rev[i][j])
    return max(0.0, kl)


def _channel_entropy_production(values: List[float], n_bins: int = _N_BINS) -> float:
    """Entropy production rate for a single channel."""
    if len(values) < 3:
        return 0.0
    bins = _discretize(values, n_bins)

    p_fwd = _transition_probs(bins, n_bins)
    reversed_bins = list(reversed(bins))
    p_rev = _transition_probs(reversed_bins, n_bins)

    # State occupation frequency
    freq = [0.0] * n_bins
    for b in bins:
        freq[b] += 1.0
    total = sum(freq)
    freq = [f / total for f in freq]

    return _kl_divergence_matrices(p_fwd, p_rev, freq, n_bins)


def compute_entropy_production_rate(trace: List[CausalEvent]) -> float:
    """Compute aggregate entropy production rate across all channels.

    Sums per-channel (marginal) entropy production. This is an upper bound
    on joint entropy production and may miss cross-channel irreversibility.
    Use ``compute_joint_entropy_production`` for joint-space measurement.

    Returns sigma > 0 for irreversible (human) traces, sigma ~ 0 for
    time-symmetric (LLM/random) traces.
    """
    if len(trace) < 3:
        return 0.0

    glucose = [e.glucose_at_event for e in trace]
    latency = [e.latency_ms for e in trace]
    complexity = [e.syntactic_complexity for e in trace]

    sigma_glucose = _channel_entropy_production(glucose)
    sigma_latency = _channel_entropy_production(latency)
    sigma_complexity = _channel_entropy_production(complexity)

    return sigma_glucose + sigma_latency + sigma_complexity


def compute_joint_entropy_production(
    trace: List[CausalEvent], n_bins: int = 5,
) -> float:
    """Compute entropy production over the joint state space of all channels.

    Uses a coarser binning (default 5) than marginal to keep the joint
    state space tractable (5^3 = 125 states vs 10^3 = 1000). Captures
    cross-channel irreversibility that marginal summation misses.
    """
    if len(trace) < 3:
        return 0.0

    glucose = [e.glucose_at_event for e in trace]
    latency = [e.latency_ms for e in trace]
    complexity = [e.syntactic_complexity for e in trace]

    g_bins = _discretize(glucose, n_bins)
    l_bins = _discretize(latency, n_bins)
    c_bins = _discretize(complexity, n_bins)

    # Encode joint state as single integer
    n_states = n_bins ** 3
    joint_bins = [g * n_bins * n_bins + l * n_bins + c for g, l, c in zip(g_bins, l_bins, c_bins)]

    # Forward transition matrix
    fwd_counts = [[0] * n_states for _ in range(n_states)]
    for t in range(len(joint_bins) - 1):
        fwd_counts[joint_bins[t]][joint_bins[t + 1]] += 1

    # Reverse transition matrix
    rev_bins = list(reversed(joint_bins))
    rev_counts = [[0] * n_states for _ in range(n_states)]
    for t in range(len(rev_bins) - 1):
        rev_counts[rev_bins[t]][rev_bins[t + 1]] += 1

    def _normalize(counts: List[List[int]]) -> List[List[float]]:
        probs = [[0.0] * n_states for _ in range(n_states)]
        for i in range(n_states):
            row_sum = sum(counts[i])
            if row_sum > 0:
                for j in range(n_states):
                    probs[i][j] = (counts[i][j] + _EPSILON) / (row_sum + n_states * _EPSILON)
            else:
                for j in range(n_states):
                    probs[i][j] = 1.0 / n_states
        return probs

    p_fwd = _normalize(fwd_counts)
    p_rev = _normalize(rev_counts)

    # State occupation frequency
    freq = [0.0] * n_states
    for b in joint_bins:
        freq[b] += 1.0
    total = sum(freq)
    freq = [f / total for f in freq]

    return _kl_divergence_matrices(p_fwd, p_rev, freq, n_states)


def _autocorrelation(values: List[float], lag: int = 1) -> float:
    """Compute lag-k autocorrelation."""
    n = len(values)
    if n < lag + 2:
        return 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values)
    if var < _EPSILON:
        return 0.0
    cov = sum((values[t] - mu) * (values[t + lag] - mu) for t in range(n - lag))
    return cov / var


def compute_time_asymmetry(trace: List[CausalEvent]) -> Dict[str, float]:
    """Per-channel time asymmetry via forward/reverse autocorrelation difference.

    Returns dict with per-channel scores and an aggregate measure.
    """
    if len(trace) < 3:
        return {
            "glucose_asymmetry": 0.0,
            "latency_asymmetry": 0.0,
            "complexity_asymmetry": 0.0,
            "aggregate_asymmetry": 0.0,
        }

    channels = {
        "glucose": [e.glucose_at_event for e in trace],
        "latency": [e.latency_ms for e in trace],
        "complexity": [e.syntactic_complexity for e in trace],
    }

    result: Dict[str, float] = {}
    scores = []
    for name, values in channels.items():
        fwd_ac = _autocorrelation(values)
        rev_ac = _autocorrelation(list(reversed(values)))
        asymmetry = abs(fwd_ac - rev_ac)
        result[f"{name}_asymmetry"] = round(asymmetry, 6)
        scores.append(asymmetry)

    result["aggregate_asymmetry"] = round(sum(scores) / len(scores), 6)
    return result


def compute_dissipation_trajectory(
    trace: List[CausalEvent], window: int = 10
) -> List[float]:
    """Sliding-window entropy production rate over the trace.

    Human traces should show increasing dissipation (fatigue ramp);
    LLM traces should be flat near zero.
    """
    if len(trace) < window or window < 3:
        return []

    trajectory = []
    for start in range(len(trace) - window + 1):
        window_trace = trace[start : start + window]
        sigma = compute_entropy_production_rate(window_trace)
        trajectory.append(round(sigma, 6))
    return trajectory
