"""Simplified Integrated Information (Phi) computation for causal traces.

Measures how much the channels of a writing trace are informationally
integrated vs independent. Based on Tononi's IIT: genuine human writing
has causally coupled channels (glucose, latency, complexity, failure)
producing high Phi, while LLM-generated traces have independent channels
producing Phi near zero.
"""
from __future__ import annotations

import math
from typing import List, Dict, Tuple

from .schema import CausalEvent

__all__ = [
    "compute_phi",
    "compute_channel_integration_matrix",
    "compute_phi_trajectory",
    "compute_phi_nbins_sensitivity",
]

N_BINS = 8
EPSILON = 1e-10


def _discretize(values: List[float], n_bins: int = N_BINS) -> List[int]:
    """Equal-width binning of continuous values into integer bin indices."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < EPSILON:
        return [0] * len(values)
    bin_width = (hi - lo) / n_bins
    return [min(n_bins - 1, int((v - lo) / bin_width)) for v in values]


def _joint_entropy(columns: List[List[int]], n_bins: int = N_BINS) -> float:
    """Compute joint entropy H(X1, X2, ...) from discretized columns using log2."""
    if not columns or not columns[0]:
        return 0.0
    n = len(columns[0])
    # Count joint occurrences using a dict keyed by tuples
    counts: Dict[tuple, int] = {}
    for i in range(n):
        key = tuple(col[i] for col in columns)
        counts[key] = counts.get(key, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > EPSILON:
            h -= p * math.log2(p)
    return h


def _extract_channels(trace: List[CausalEvent]) -> Dict[str, List[float]]:
    """Extract the 4 channels from a causal trace."""
    return {
        "glucose": [e.glucose_at_event for e in trace],
        "latency": [e.latency_ms for e in trace],
        "complexity": [e.syntactic_complexity for e in trace],
        "failure": [1.0 if e.status != "success" else 0.0 for e in trace],
    }


def _all_bipartitions(names: List[str]) -> List[Tuple[List[str], List[str]]]:
    """Generate all non-trivial bipartitions of channel names.

    For 4 channels there are 2^(4-1) - 1 = 7 non-trivial bipartitions.
    A bipartition splits names into two non-empty groups.
    We enumerate subsets of size 1..n//2 to avoid duplicates.
    """
    n = len(names)
    results = []
    # Enumerate all subsets via bitmask, keeping only one of each complementary pair
    for mask in range(1, 2**n - 1):
        complement = ((2**n) - 1) ^ mask
        if mask < complement:
            group_a = [names[i] for i in range(n) if mask & (1 << i)]
            group_b = [names[i] for i in range(n) if not (mask & (1 << i))]
            results.append((group_a, group_b))
    return results


def compute_phi(trace: List[CausalEvent]) -> float:
    """Compute simplified Integrated Information (Phi) for a causal trace.

    Phi = minimum mutual information across all bipartitions of the 4 channels
    (glucose, latency, complexity, failure), normalized to [0, 1].

    For each bipartition into groups A and B:
        MI(A; B) = H(A) + H(B) - H(A, B)
    Phi = min over all bipartitions of MI(A; B).

    Returns 0.0 for traces too short to compute meaningful entropy.
    """
    if len(trace) < 3:
        return 0.0

    raw = _extract_channels(trace)
    channel_names = list(raw.keys())

    # Discretize each channel
    discretized = {name: _discretize(raw[name]) for name in channel_names}

    # Joint entropy of all 4 channels
    all_cols = [discretized[name] for name in channel_names]
    h_joint = _joint_entropy(all_cols)

    if h_joint < EPSILON:
        return 0.0

    bipartitions = _all_bipartitions(channel_names)

    min_mi = float("inf")
    for group_a, group_b in bipartitions:
        cols_a = [discretized[name] for name in group_a]
        cols_b = [discretized[name] for name in group_b]
        h_a = _joint_entropy(cols_a)
        h_b = _joint_entropy(cols_b)
        mi = h_a + h_b - h_joint
        min_mi = min(min_mi, mi)

    if min_mi == float("inf") or min_mi < EPSILON:
        return 0.0

    # Normalize by sum of marginal entropies (H_indep) for length-invariant [0, 1] range.
    # H_indep is an upper bound on joint entropy and stable across trace lengths,
    # unlike H_joint which shrinks with fewer samples.
    h_indep = sum(_joint_entropy([discretized[name]]) for name in channel_names)
    normalizer = h_indep if h_indep > EPSILON else h_joint
    phi = min(1.0, max(0.0, min_mi / normalizer))
    return round(phi, 4)


def compute_channel_integration_matrix(trace: List[CausalEvent]) -> Dict[str, Dict[str, float]]:
    """Compute pairwise mutual information between all channel pairs.

    Returns a 4x4 matrix (dict of dicts) showing which channels are most
    integrated. Human traces should show high MI for glucose-latency and
    failure-complexity; LLM traces should show low MI across all pairs.
    """
    if len(trace) < 3:
        names = ["glucose", "latency", "complexity", "failure"]
        return {a: {b: 0.0 for b in names} for a in names}

    raw = _extract_channels(trace)
    channel_names = list(raw.keys())
    discretized = {name: _discretize(raw[name]) for name in channel_names}

    matrix: Dict[str, Dict[str, float]] = {}
    for a in channel_names:
        matrix[a] = {}
        for b in channel_names:
            if a == b:
                # Self-information: entropy of the channel
                h = _joint_entropy([discretized[a]])
                matrix[a][b] = round(h, 4)
            else:
                h_a = _joint_entropy([discretized[a]])
                h_b = _joint_entropy([discretized[b]])
                h_ab = _joint_entropy([discretized[a], discretized[b]])
                mi = max(0.0, h_a + h_b - h_ab)
                matrix[a][b] = round(mi, 4)
    return matrix


def compute_phi_trajectory(trace: List[CausalEvent], window: int = 15) -> List[float]:
    """Compute Phi over a sliding window across the trace.

    Shows how integration evolves over the writing session.
    Human traces should maintain or increase Phi; LLM traces should
    show low/erratic Phi.
    """
    if len(trace) < window or window < 3:
        return []
    return [compute_phi(trace[i:i + window]) for i in range(len(trace) - window + 1)]


def compute_phi_nbins_sensitivity(
    trace: List[CausalEvent],
    bin_values: List[int] | None = None,
) -> Dict[str, float]:
    """Compute Phi across multiple N_BINS values to assess binning sensitivity.

    Returns dict mapping bin count to Phi value.
    Reports coefficient of variation (CV) — if CV > 0.10, the result
    is sensitive to binning and should be noted in the paper.
    """
    if bin_values is None:
        bin_values = [4, 6, 8, 10, 12, 16]
    if len(trace) < 3:
        return {"phi_by_bins": {str(b): 0.0 for b in bin_values}, "cv": 0.0, "sensitive": False}

    raw = _extract_channels(trace)
    channel_names = list(raw.keys())
    bipartitions = _all_bipartitions(channel_names)

    results: Dict[str, float] = {}
    for nb in bin_values:
        discretized = {name: _discretize(raw[name], n_bins=nb) for name in channel_names}
        all_cols = [discretized[name] for name in channel_names]
        h_joint = _joint_entropy(all_cols, n_bins=nb)
        if h_joint < EPSILON:
            results[str(nb)] = 0.0
            continue
        min_mi = float("inf")
        for group_a, group_b in bipartitions:
            cols_a = [discretized[name] for name in group_a]
            cols_b = [discretized[name] for name in group_b]
            h_a = _joint_entropy(cols_a, n_bins=nb)
            h_b = _joint_entropy(cols_b, n_bins=nb)
            mi = h_a + h_b - h_joint
            min_mi = min(min_mi, mi)
        h_indep = sum(_joint_entropy([discretized[name]], n_bins=nb) for name in channel_names)
        normalizer = h_indep if h_indep > EPSILON else h_joint
        phi = min(1.0, max(0.0, min_mi / normalizer))
        results[str(nb)] = round(phi, 4)

    phi_vals = list(results.values())
    mean_phi = sum(phi_vals) / len(phi_vals) if phi_vals else 0.0
    variance = sum((v - mean_phi) ** 2 for v in phi_vals) / len(phi_vals) if phi_vals else 0.0
    cv = (variance ** 0.5) / mean_phi if mean_phi > EPSILON else 0.0

    return {
        "phi_by_bins": results,
        "mean": round(mean_phi, 4),
        "cv": round(cv, 4),
        "sensitive": cv > 0.10,
    }
