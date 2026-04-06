"""Temporal binding index for measuring long-range temporal coherence in causal traces.

Human writing exhibits fat-tailed (power-law) decay of mutual information across
time lags — the embodied cognitive state carries forward. LLM traces show thin-tailed
(exponential) decay — no persistent cognitive state binds events across time.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Dict

from .schema import CausalEvent

__all__ = [
    "compute_temporal_binding_index",
    "compute_binding_decay_curve",
    "compute_decay_exponent",
    "compute_cross_channel_binding",
]

EPSILON = 1e-10
NUM_BINS = 8
CHANNELS = ("glucose", "latency", "complexity", "failure")


def _extract_channels(trace: List[CausalEvent]) -> Dict[str, List[float]]:
    """Extract multivariate state channels from a causal trace."""
    return {
        "glucose": [e.glucose_at_event for e in trace],
        "latency": [e.latency_ms for e in trace],
        "complexity": [e.syntactic_complexity for e in trace],
        "failure": [1.0 if e.status != "success" else 0.0 for e in trace],
    }


def _discretize(values: List[float], num_bins: int = NUM_BINS) -> List[int]:
    """Equal-width binning into num_bins discrete bins."""
    lo = min(values)
    hi = max(values)
    if hi - lo < EPSILON:
        return [0] * len(values)
    width = (hi - lo) / num_bins
    return [min(num_bins - 1, int((v - lo) / width)) for v in values]


def _mutual_information(x: List[float], y: List[float]) -> float:
    """Compute MI(X; Y) in bits using discretized joint histograms."""
    n = len(x)
    if n < 2:
        return 0.0

    bx = _discretize(x)
    by = _discretize(y)

    # Joint and marginal counts
    joint: Dict[Tuple[int, int], int] = {}
    mx: Dict[int, int] = {}
    my: Dict[int, int] = {}
    for i in range(n):
        pair = (bx[i], by[i])
        joint[pair] = joint.get(pair, 0) + 1
        mx[bx[i]] = mx.get(bx[i], 0) + 1
        my[by[i]] = my.get(by[i], 0) + 1

    mi = 0.0
    for (a, b), count in joint.items():
        p_ab = count / n
        p_a = mx[a] / n
        p_b = my[b] / n
        mi += p_ab * math.log2(p_ab / (p_a * p_b + EPSILON))

    return max(0.0, mi)


def _multivariate_mi_at_lag(channels: Dict[str, List[float]], lag: int) -> float:
    """Average MI across all channels at a given lag."""
    n = len(next(iter(channels.values())))
    if lag >= n:
        return 0.0

    total_mi = 0.0
    count = 0
    for name in CHANNELS:
        vals = channels[name]
        x = vals[:n - lag]
        y = vals[lag:]
        mi = _mutual_information(x, y)
        total_mi += mi
        count += 1

    return total_mi / count if count > 0 else 0.0


def _effective_max_lag(trace_len: int, max_lag: int) -> int:
    """Auto-adjust max_lag to ensure enough data points."""
    return min(max_lag, trace_len // 3)


def compute_temporal_binding_index(trace: List[CausalEvent], max_lag: int = 20) -> float:
    """Compute the temporal binding index — integral of MI across lags, normalized by MI(1).

    The normalization converts raw MI sums into a scale-invariant measure:
    each lag's MI is divided by MI(1), then summed. A slowly-decaying signal
    yields TBI ~ max_lag; a rapidly-decaying signal yields TBI << max_lag.
    The result is further scaled by MI(1) so that traces with higher absolute
    coherence score higher than noise floors.
    """
    if len(trace) < 3:
        return 0.0

    channels = _extract_channels(trace)
    effective_lag = _effective_max_lag(len(trace), max_lag)
    if effective_lag < 1:
        return 0.0

    mi_values = []
    for k in range(1, effective_lag + 1):
        mi_values.append(_multivariate_mi_at_lag(channels, k))

    mi_at_1 = mi_values[0]
    if mi_at_1 < EPSILON:
        return 0.0

    # Raw sum of MI across all lags captures both absolute coherence
    # strength (via MI magnitudes) and persistence (via number of
    # non-negligible lags). Equivalent to sum(MI(k)/MI(1)) * MI(1).
    return sum(mi_values)


def compute_binding_decay_curve(
    trace: List[CausalEvent], max_lag: int = 20
) -> List[Tuple[int, float]]:
    """Return list of (lag, mutual_information) pairs."""
    if len(trace) < 3:
        return []

    channels = _extract_channels(trace)
    effective_lag = _effective_max_lag(len(trace), max_lag)
    if effective_lag < 1:
        return []

    result = []
    for k in range(1, effective_lag + 1):
        mi = _multivariate_mi_at_lag(channels, k)
        result.append((k, mi))
    return result


def _ols_fit(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Simple OLS for y = slope*x + intercept. Returns (slope, intercept, R^2)."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sxx - sx * sx
    if abs(denom) < EPSILON:
        return 0.0, sy / n if n > 0 else 0.0, 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    # R^2
    y_mean = sy / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - slope * xi - intercept) ** 2 for xi, yi in zip(x, y))

    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0
    return slope, intercept, max(0.0, r2)


def compute_decay_exponent(
    trace: List[CausalEvent], max_lag: int = 20
) -> Dict[str, float]:
    """Fit power-law and exponential decay to the MI curve."""
    curve = compute_binding_decay_curve(trace, max_lag)
    if len(curve) < 3:
        return {
            "power_law_exponent": 0.0,
            "exponential_rate": 0.0,
            "power_law_r2": 0.0,
            "exponential_r2": 0.0,
            "decay_type": "exponential",
        }

    # Filter out zero-MI points for log fitting; require >=8 for reliable model selection
    valid = [(k, mi) for k, mi in curve if mi > EPSILON]
    if len(valid) < 8:
        return {
            "power_law_exponent": 0.0,
            "exponential_rate": 0.0,
            "power_law_r2": 0.0,
            "exponential_r2": 0.0,
            "decay_type": "exponential",
        }

    lags = [float(k) for k, _ in valid]
    mis = [mi for _, mi in valid]
    log_lags = [math.log(k) for k in lags]
    log_mis = [math.log(mi) for mi in mis]

    # Power-law: log(MI) = -alpha * log(k) + c
    pl_slope, _, pl_r2 = _ols_fit(log_lags, log_mis)
    alpha = -pl_slope

    # Exponential: log(MI) = -lambda * k + c
    exp_slope, _, exp_r2 = _ols_fit(lags, log_mis)
    lam = -exp_slope

    # Require meaningful R² difference to avoid spurious classification
    if abs(pl_r2 - exp_r2) < 0.02:
        decay_type = "inconclusive"
    elif pl_r2 > exp_r2:
        decay_type = "power_law"
    else:
        decay_type = "exponential"

    return {
        "power_law_exponent": round(alpha, 6),
        "exponential_rate": round(lam, 6),
        "power_law_r2": round(pl_r2, 6),
        "exponential_r2": round(exp_r2, 6),
        "decay_type": decay_type,
    }


def compute_cross_channel_binding(
    trace: List[CausalEvent], lag: int = 5
) -> Dict[str, float]:
    """Compute cross-channel MI at a given lag for all channel pairs."""
    if len(trace) < lag + 2:
        return {}

    channels = _extract_channels(trace)
    n = len(trace)
    result: Dict[str, float] = {}

    channel_names = list(CHANNELS)
    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            a_name = channel_names[i]
            b_name = channel_names[j]
            x = channels[a_name][:n - lag]
            y = channels[b_name][lag:]
            mi = _mutual_information(x, y)
            result[f"{a_name}->{b_name}"] = round(mi, 6)

    return result
