"""Free energy trajectory analysis for active inference detection.

Under Friston's Free Energy Principle, biological systems minimize prediction
error about their own actions. A human writer has an implicit self-model that
produces characteristic trajectories: warmup (decreasing error), plateau
(stable performance), fatigue (increasing error). LLM traces lack this
structure entirely.
"""
from __future__ import annotations

from typing import List, Dict, Optional

from .schema import CausalEvent

__all__ = [
    "compute_prediction_error",
    "compute_free_energy_trajectory",
    "compute_surprise_spikes",
    "compute_adaptation_rate",
    "compute_phenomenological_gap",
    "compute_shape_score",
]


def _ols_slope_intercept(x: List[float], y: List[float]) -> tuple:
    """Simple OLS: y = a + b*x. Returns (intercept, slope)."""
    n = len(x)
    if n < 2:
        return (0.0, 0.0)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return (sy / n if n > 0 else 0.0, 0.0)
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return (a, b)


def _residual_sum_of_squares(x: List[float], y: List[float]) -> float:
    """RSS from OLS fit."""
    a, b = _ols_slope_intercept(x, y)
    return sum((yi - a - b * xi) ** 2 for xi, yi in zip(x, y))


def _moving_average(values: List[float], window: int) -> List[float]:
    """Simple moving average. Output length = len(values) - window + 1."""
    if len(values) < window or window < 1:
        return list(values)
    result = []
    running = sum(values[:window])
    result.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        result.append(running / window)
    return result


def compute_prediction_error(trace: List[CausalEvent]) -> List[float]:
    """Compute per-event prediction error (surprise).

    error[t] = syntactic_complexity[t] * (1 if failure else 0)
             + abs(glucose_change[t]) * 2

    High-complexity failures are more surprising; rapid glucose changes
    indicate unexpected resource demands.
    """
    if not trace:
        return []

    errors = []
    for i, event in enumerate(trace):
        failed = 1.0 if event.failure_mode is not None else 0.0
        complexity_surprise = event.syntactic_complexity * failed

        # Glucose change: use difference from previous event
        if i > 0:
            glucose_change = abs(event.glucose_at_event - trace[i - 1].glucose_at_event)
        else:
            glucose_change = 0.0

        errors.append(complexity_surprise + glucose_change * 2.0)

    return errors


def compute_free_energy_trajectory(
    trace: List[CausalEvent], window: int = 10
) -> dict:
    """Compute smoothed prediction error and fit a 3-phase model.

    Phases: warmup (decreasing error), plateau (stable), fatigue (increasing).
    Tries all possible split points and picks the partition that minimizes
    total residual. For traces > 100 events, samples every 5 split points.
    """
    if not trace:
        return {
            "trajectory": [],
            "warmup_slope": 0.0,
            "plateau_level": 0.0,
            "fatigue_slope": 0.0,
            "trajectory_score": 0.0,
            "phase_boundaries": [0, 0],
        }

    raw_errors = compute_prediction_error(trace)
    smoothed = _moving_average(raw_errors, min(window, len(raw_errors)))

    if len(smoothed) < 3:
        mean_val = sum(smoothed) / len(smoothed) if smoothed else 0.0
        return {
            "trajectory": smoothed,
            "warmup_slope": 0.0,
            "plateau_level": mean_val,
            "fatigue_slope": 0.0,
            "trajectory_score": 0.0,
            "phase_boundaries": [0, len(smoothed)],
        }

    n = len(smoothed)
    x_all = list(range(n))

    # Determine split point step size
    step = 5 if n > 100 else 1
    min_segment = max(2, n // 10)

    best_rss = float("inf")
    best_b1 = min_segment
    best_b2 = n - min_segment

    for b1 in range(min_segment, n - 2 * min_segment + 1, step):
        for b2 in range(b1 + min_segment, n - min_segment + 1, step):
            rss1 = _residual_sum_of_squares(x_all[:b1], smoothed[:b1])
            rss2 = _residual_sum_of_squares(x_all[b1:b2], smoothed[b1:b2])
            rss3 = _residual_sum_of_squares(x_all[b2:], smoothed[b2:])
            total_rss = rss1 + rss2 + rss3
            if total_rss < best_rss:
                best_rss = total_rss
                best_b1 = b1
                best_b2 = b2

    # Compute phase statistics
    _, warmup_slope = _ols_slope_intercept(x_all[:best_b1], smoothed[:best_b1])
    plateau_vals = smoothed[best_b1:best_b2]
    plateau_level = sum(plateau_vals) / len(plateau_vals) if plateau_vals else 0.0
    _, fatigue_slope = _ols_slope_intercept(x_all[best_b2:], smoothed[best_b2:])

    # R² for 3-phase model vs flat baseline (1-phase)
    mean_val = sum(smoothed) / n
    total_variance = sum((v - mean_val) ** 2 for v in smoothed)
    if total_variance > 1e-15:
        r2_3phase = max(0.0, 1.0 - best_rss / total_variance)
    else:
        r2_3phase = 0.0

    # R² for 1-phase (linear) baseline
    rss_linear = _residual_sum_of_squares(x_all, smoothed)
    r2_linear = max(0.0, 1.0 - rss_linear / total_variance) if total_variance > 1e-15 else 0.0

    # trajectory_score: improvement of 3-phase over linear baseline
    trajectory_score = max(0.0, r2_3phase - r2_linear)

    return {
        "trajectory": smoothed,
        "warmup_slope": round(warmup_slope, 6),
        "plateau_level": round(plateau_level, 6),
        "fatigue_slope": round(fatigue_slope, 6),
        "trajectory_score": round(trajectory_score, 6),
        "r2_3phase": round(r2_3phase, 6),
        "r2_linear": round(r2_linear, 6),
        "phase_boundaries": [best_b1, best_b2],
    }


def compute_shape_score(trace: List[CausalEvent]) -> float:
    """Score whether free energy trajectory has the expected biological shape.

    FEP predicts: warmup (negative slope = decreasing surprise as writer
    calibrates), then fatigue (positive slope = increasing surprise as
    resources deplete). The SHAPE matters, not just the R² fit quality.

    Returns: product of |warmup_slope| * fatigue_slope, clamped to [0, 1].
    Positive only when warmup is negative AND fatigue is positive.
    """
    result = compute_free_energy_trajectory(trace)
    if not isinstance(result, dict):
        return 0.0
    warmup = result.get("warmup_slope", 0.0)
    fatigue = result.get("fatigue_slope", 0.0)
    score = result.get("trajectory_score", 0.0)
    # Expected: warmup < 0 (decreasing), fatigue > 0 (increasing)
    if warmup < 0 and fatigue > 0:
        shape = min(1.0, abs(warmup) * fatigue * 100.0) * score
        return max(0.0, min(1.0, shape))
    # Partial credit: just fatigue positive (common in short traces)
    elif fatigue > 0:
        return max(0.0, min(1.0, fatigue * 10.0 * score * 0.5))
    return 0.0


def compute_surprise_spikes(
    trace: List[CausalEvent], threshold: float = 2.0
) -> List[dict]:
    """Detect sudden spikes in prediction error (> threshold * mean).

    In human traces, spikes cluster at content boundaries.
    In LLM traces, spikes are random or absent.
    """
    errors = compute_prediction_error(trace)
    if not errors:
        return []

    mean_error = sum(errors) / len(errors)
    if mean_error < 1e-15:
        return []

    cutoff = threshold * mean_error
    spikes = []
    for i, err in enumerate(errors):
        if err > cutoff:
            context = trace[i].intention if i < len(trace) else ""
            spikes.append({
                "index": i,
                "magnitude": round(err / mean_error, 4),
                "context": context,
            })

    return spikes


def compute_adaptation_rate(trace: List[CausalEvent]) -> float:
    """Measure post-failure complexity recovery speed.

    For each failure, track how quickly the writer returns to pre-failure
    complexity levels. Human writers show learning-curve recovery (positive
    adaptation rate). LLM traces show no adaptation (rate near zero).
    """
    if not trace:
        return 0.0

    recovery_rates: List[float] = []
    lookahead = 5  # Check next N events after failure

    for i, event in enumerate(trace):
        if event.failure_mode is None:
            continue

        # Pre-failure complexity: average of up to 3 preceding successful events
        pre_complexities = []
        for j in range(max(0, i - 3), i):
            if trace[j].failure_mode is None:
                pre_complexities.append(trace[j].syntactic_complexity)
        if not pre_complexities:
            continue
        pre_level = sum(pre_complexities) / len(pre_complexities)

        # Post-failure recovery: track complexity of next successful events
        post_complexities = []
        for j in range(i + 1, min(len(trace), i + 1 + lookahead)):
            if trace[j].failure_mode is None:
                post_complexities.append(trace[j].syntactic_complexity)

        if len(post_complexities) < 2:
            continue

        # Measure recovery: how quickly does post-failure complexity approach pre_level?
        # Use slope of (post_complexity / pre_level) over the recovery window
        ratios = [c / pre_level if pre_level > 0 else 0.0 for c in post_complexities]
        x_vals = list(range(len(ratios)))
        _, slope = _ols_slope_intercept(x_vals, ratios)
        recovery_rates.append(slope)

    if not recovery_rates:
        return 0.0

    return sum(recovery_rates) / len(recovery_rates)


def compute_phenomenological_gap(trace: List[CausalEvent]) -> float:
    """Detect subjective vs objective difficulty divergence.

    Regresses failure_rate on syntactic_complexity. The residual variance
    captures cases where the writer struggles with "easy" words or flows
    through "hard" ones — embodied state matters beyond task demands.

    Returns residual_variance / total_variance. Higher = more phenomenological gap.
    """
    if len(trace) < 5:
        return 0.0

    complexities = [e.syntactic_complexity for e in trace]
    failures = [1.0 if e.failure_mode is not None else 0.0 for e in trace]

    # Total variance of failure indicator
    mean_f = sum(failures) / len(failures)
    total_var = sum((f - mean_f) ** 2 for f in failures)
    if total_var < 1e-15:
        return 0.0

    # Regress failure on complexity
    a, b = _ols_slope_intercept(complexities, failures)
    residual_var = sum((fi - a - b * ci) ** 2 for fi, ci in zip(failures, complexities))

    return round(residual_var / total_var, 6)
