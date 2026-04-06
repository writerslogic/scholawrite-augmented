"""Consciousness-correlate signatures for distinguishing human from machine authorship.

Uses empirically validated signals grounded in consciousness science:
1. Temporal binding decay type (power-law vs exponential MI decay)
2. Causal DAG fingerprint (Granger causality across channel pairs)
3. Cross-channel integration (lagged MI between glucose and latency)
4. Free energy trajectory (3-phase prediction error structure)
5. Adaptation rate (post-failure complexity recovery)

Each signal is selected because it measures a STRUCTURAL property of
conscious cognitive processing — not just noise or variability — that
adversarial forgeries cannot easily replicate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean

from .schema import CausalEvent
from .thermodynamic import compute_entropy_production_rate
from .integrated_information import compute_phi
from .temporal_binding import (
    compute_temporal_binding_index,
    compute_decay_exponent,
    compute_cross_channel_binding,
)
from .free_energy import (
    compute_free_energy_trajectory,
    compute_phenomenological_gap,
    compute_adaptation_rate,
    compute_shape_score,
)

__all__ = [
    "ConsciousnessSignatureResult",
    "compute_consciousness_signatures",
    "compare_consciousness_profiles",
    "calibrate_normalization_ceilings",
]

# Composite score threshold for "human-like"
_HUMAN_THRESHOLD = 0.35

# Signal weights — empirically tuned by discrimination power
_WEIGHTS = {
    "causal_dag": 0.20,            # Granger mass — total causal structure
    "causal_concentration": 0.15,  # Expected-arrow ratio — DAG shape
    "cross_channel_mi": 0.15,     # glucose->latency MI — physiological coupling
    "decay_type": 0.20,           # power-law vs exponential — theoretical signal
    "free_energy": 0.15,          # 3-phase trajectory
    "adaptation": 0.15,           # post-failure recovery — directional signal
}

# Default normalization ceilings (used when no calibration data available)
_DEFAULT_CEILINGS = {
    "causal_dag": 10.0,       # F-stat total (human traces ~4, forged ~0.02)
    "cross_channel_mi": 1.5,  # MI bits
    "decay_type": 0.5,        # R² difference
    "adaptation": 0.15,       # recovery slope (human ~0.07, forged ~-0.01)
}

# Active ceilings (overridden by calibrate_normalization_ceilings)
_ceilings: Dict[str, float] = dict(_DEFAULT_CEILINGS)


def calibrate_normalization_ceilings(
    authentic_traces: List[List[CausalEvent]],
    percentile: float = 95.0,
) -> Dict[str, float]:
    """Compute normalization ceilings from authentic trace data.

    Uses the given percentile of each signal's distribution on authentic
    traces as the ceiling. Updates the module-level ceilings for subsequent
    calls to compute_consciousness_signatures().

    Returns the calibrated ceiling dict.
    """
    if not authentic_traces:
        return dict(_DEFAULT_CEILINGS)

    dag_vals, mi_vals, decay_vals, adapt_vals = [], [], [], []
    for trace in authentic_traces:
        if len(trace) < 5:
            continue
        dag_mass, _ = _compute_causal_dag_mass(trace)
        dag_vals.append(dag_mass)
        mi_vals.append(_compute_cross_channel_mi(trace))
        decay_vals.append(max(0.0, _compute_decay_type_score(trace)))
        adapt_vals.append(max(0.0, compute_adaptation_rate(trace)))

    def _percentile(vals: List[float], pct: float) -> float:
        if not vals:
            return 1.0
        s = sorted(vals)
        idx = min(len(s) - 1, int(len(s) * pct / 100.0))
        return max(s[idx], 1e-6)

    calibrated = {
        "causal_dag": round(_percentile(dag_vals, percentile), 4),
        "cross_channel_mi": round(_percentile(mi_vals, percentile), 4),
        "decay_type": round(_percentile(decay_vals, percentile), 4),
        "adaptation": round(_percentile(adapt_vals, percentile), 4),
    }

    global _ceilings
    _ceilings = calibrated
    return calibrated


@dataclass(frozen=True)
class ConsciousnessSignatureResult:
    """Aggregated consciousness-correlate signature for a causal trace."""
    entropy_production: float
    integrated_information: float
    temporal_binding: float
    free_energy_score: float
    phenomenological_gap: float
    composite_consciousness_score: float
    is_human_like: bool
    signal_breakdown: Dict[str, Any] = field(default_factory=dict)


def _normalize(value: float, ceiling: float) -> float:
    """Clamp value to [0, 1] by dividing by ceiling."""
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, value / ceiling))


def _extract_free_energy_score(trace: List[CausalEvent]) -> float:
    """Extract the trajectory_score from free energy analysis."""
    result = compute_free_energy_trajectory(trace)
    if isinstance(result, dict):
        return result.get("trajectory_score", 0.0)
    return 0.0


def _extract_free_energy_shape(trace: List[CausalEvent]) -> float:
    """Score whether free energy trajectory has the expected biological shape.

    Delegates to free_energy.compute_shape_score() (single source of truth).
    """
    return compute_shape_score(trace)


def _granger_pair(trace: List[CausalEvent], x_channel: str, y_channel: str) -> float:
    """Lag-1 Granger causality F-statistic: does x help predict y beyond y's own lag?"""
    if len(trace) < 5:
        return 0.0

    channels = {
        "glucose": [e.glucose_at_event for e in trace],
        "latency": [e.latency_ms for e in trace],
        "complexity": [e.syntactic_complexity for e in trace],
        "failure": [1.0 if e.failure_mode else 0.0 for e in trace],
    }
    x = channels[x_channel]
    y = channels[y_channel]
    n = len(x) - 1
    if n < 4:
        return 0.0

    # Zero-variance channels have no causal structure to detect
    y_var = sum((v - sum(y) / len(y)) ** 2 for v in y)
    x_var = sum((v - sum(x) / len(x)) ** 2 for v in x)
    if y_var < 1e-10 or x_var < 1e-10:
        return 0.0

    y_lag = y[:-1]
    y_cur = y[1:]
    x_lag = x[:-1]

    mu_yl = mean(y_lag)
    mu_yc = mean(y_cur)

    # Restricted model: y[t] = a + b*y[t-1]
    num_r = sum((yl - mu_yl) * (yc - mu_yc) for yl, yc in zip(y_lag, y_cur))
    den_r = sum((yl - mu_yl) ** 2 for yl in y_lag)
    b_r = num_r / den_r if den_r > 1e-15 else 0.0
    a_r = mu_yc - b_r * mu_yl
    rss_r = sum((yc - a_r - b_r * yl) ** 2 for yc, yl in zip(y_cur, y_lag))

    # Unrestricted: y[t] = a + b*y[t-1] + c*x[t-1]
    mu_xl = mean(x_lag)
    X = list(zip(y_lag, x_lag))
    XtX = [[sum(xi[j] * xi[k] for xi in X) for k in range(2)] for j in range(2)]
    XtY = [sum(xi[j] * yc for xi, yc in zip(X, y_cur)) for j in range(2)]
    det = XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0]
    if abs(det) < 1e-15:
        return 0.0
    b0 = (XtY[0] * XtX[1][1] - XtY[1] * XtX[0][1]) / det
    b1 = (XtX[0][0] * XtY[1] - XtX[1][0] * XtY[0]) / det
    a_u = mu_yc - b0 * mu_yl - b1 * mu_xl
    rss_u = sum(
        (yc - a_u - b0 * yl - b1 * xl) ** 2
        for yc, yl, xl in zip(y_cur, y_lag, x_lag)
    )

    if rss_u < 1e-15:
        return 100.0
    f_stat = ((rss_r - rss_u) / 1) / (rss_u / (n - 3)) if n > 3 else 0.0
    return max(0.0, f_stat)


def _compute_causal_dag_mass(trace: List[CausalEvent]) -> Tuple[float, Dict[str, float]]:
    """Total Granger causality F-statistic across all directed channel pairs.

    Human traces show strong causal structure (failure→glucose, glucose→latency).
    Forged traces show near-zero Granger mass — channels are causally independent.
    """
    channels = ["glucose", "latency", "complexity", "failure"]
    dag = {}
    total = 0.0
    for x in channels:
        for y in channels:
            if x != y:
                f = _granger_pair(trace, x, y)
                dag[f"{x}->{y}"] = round(f, 4)
                total += f
    return total, dag


def _compute_causal_concentration(dag: Dict[str, float]) -> float:
    """How concentrated is causal mass in theoretically expected arrows?

    Human writing should concentrate Granger causality in specific directions:
    failure→glucose (resource depletion from repair), latency→failure
    (slow responses predict subsequent failures). The RATIO of expected-arrow
    mass to total mass measures whether the causal DAG has the right shape,
    not just the right magnitude.
    """
    expected_arrows = ["failure->glucose", "latency->failure", "complexity->glucose"]
    expected_mass = sum(dag.get(a, 0.0) for a in expected_arrows)
    total = sum(dag.values())
    if total < 0.01:
        return 0.0
    return expected_mass / total


def _compute_cross_channel_mi(trace: List[CausalEvent]) -> float:
    """Glucose→latency cross-channel MI at lag 3.

    Human writing couples metabolic state to motor output with a temporal lag.
    Forged traces lack this physiological coupling.
    """
    binding = compute_cross_channel_binding(trace, lag=3)
    return binding.get("glucose->latency", 0.0)


def _compute_decay_type_score(trace: List[CausalEvent]) -> float:
    """Power-law R² minus exponential R² from MI decay curve.

    Positive = power-law (fat-tailed, human-like persistent binding).
    Negative = exponential (thin-tailed, Markov-like machine generation).
    """
    decay = compute_decay_exponent(trace)
    return decay["power_law_r2"] - decay["exponential_r2"]


def compute_consciousness_signatures(
    trace: List[CausalEvent],
    weights: Optional[Dict[str, float]] = None,
) -> ConsciousnessSignatureResult:
    """Compute consciousness-correlate measurements and combine into composite score.

    Uses 6 signals weighted by discrimination power (see _WEIGHTS):
    - Causal DAG mass (20%): total Granger causality across channel pairs
    - Causal concentration (15%): expected-arrow ratio — DAG shape
    - Cross-channel MI (15%): glucose→latency physiological coupling
    - Decay type (20%): power-law vs exponential MI decay
    - Free energy (15%): 3-phase prediction error trajectory
    - Adaptation rate (15%): post-failure complexity recovery
    """
    if not trace or len(trace) < 3:
        return ConsciousnessSignatureResult(
            entropy_production=0.0,
            integrated_information=0.0,
            temporal_binding=0.0,
            free_energy_score=0.0,
            phenomenological_gap=0.0,
            composite_consciousness_score=0.0,
            is_human_like=False,
            signal_breakdown={},
        )

    # === Legacy raw measurements (kept for backward compat / analysis) ===
    entropy_prod = compute_entropy_production_rate(trace)
    phi = compute_phi(trace)
    tbi = compute_temporal_binding_index(trace)
    fe_score = _extract_free_energy_score(trace)
    fe_shape = _extract_free_energy_shape(trace)
    phenom_gap = compute_phenomenological_gap(trace)

    # === Discriminative signals for composite ===
    causal_dag_mass, dag_detail = _compute_causal_dag_mass(trace)
    causal_conc = _compute_causal_concentration(dag_detail)
    cross_mi = _compute_cross_channel_mi(trace)
    decay_score = _compute_decay_type_score(trace)
    adapt_rate = compute_adaptation_rate(trace)

    # Normalize each signal to [0, 1] using calibrated ceilings
    c = _ceilings
    norm_dag = _normalize(causal_dag_mass, c["causal_dag"])
    norm_conc = causal_conc  # already [0, 1] (ratio)
    norm_mi = _normalize(cross_mi, c["cross_channel_mi"])
    norm_decay = _normalize(max(0.0, decay_score), c["decay_type"])
    norm_fe = fe_shape  # already [0, 1]
    norm_adapt = _normalize(max(0.0, adapt_rate), c["adaptation"])

    # Weighted composite (use caller-provided weights or defaults)
    w = weights if weights is not None else _WEIGHTS
    composite = (
        w.get("causal_dag", 0.0) * norm_dag
        + w.get("causal_concentration", 0.0) * norm_conc
        + w.get("cross_channel_mi", 0.0) * norm_mi
        + w.get("decay_type", 0.0) * norm_decay
        + w.get("free_energy", 0.0) * norm_fe
        + w.get("adaptation", 0.0) * norm_adapt
    )

    signal_breakdown = {
        "raw": {
            "entropy_production": entropy_prod,
            "integrated_information": phi,
            "temporal_binding": tbi,
            "free_energy_score": fe_score,
            "phenomenological_gap": phenom_gap,
            "free_energy_shape": round(fe_shape, 6),
            "causal_dag_mass": round(causal_dag_mass, 4),
            "causal_concentration": round(causal_conc, 4),
            "cross_channel_mi": round(cross_mi, 6),
            "decay_type_score": round(decay_score, 6),
            "adaptation_rate": round(adapt_rate, 6),
        },
        "normalized": {
            "causal_dag": round(norm_dag, 4),
            "causal_concentration": round(norm_conc, 4),
            "cross_channel_mi": round(norm_mi, 4),
            "decay_type": round(norm_decay, 4),
            "free_energy": round(norm_fe, 4),
            "adaptation": round(norm_adapt, 4),
        },
        "weights": dict(w),
        "causal_dag_detail": dag_detail,
    }

    return ConsciousnessSignatureResult(
        entropy_production=entropy_prod,
        integrated_information=phi,
        temporal_binding=tbi,
        free_energy_score=fe_score,
        phenomenological_gap=phenom_gap,
        composite_consciousness_score=round(composite, 4),
        is_human_like=composite > _HUMAN_THRESHOLD,
        signal_breakdown=signal_breakdown,
    )


def compare_consciousness_profiles(
    human_traces: List[List[CausalEvent]],
    machine_traces: List[List[CausalEvent]],
    machine_traces_by_tier: Optional[Dict[str, List[List[CausalEvent]]]] = None,
) -> Dict[str, Any]:
    """Compare consciousness signatures between human and machine trace groups.

    Computes per-metric AUC (discrimination power) and aggregate statistics.

    When machine_traces_by_tier is provided, also computes a signal × tier
    AUC breakdown showing how each signal performs against each adversary tier.
    """
    from .metrics import auc as compute_auc

    human_results = [compute_consciousness_signatures(t) for t in human_traces]
    machine_results = [compute_consciousness_signatures(t) for t in machine_traces]

    if not human_results or not machine_results:
        return {
            "per_metric_auc": {},
            "composite_auc": 0.0,
            "human_stats": {},
            "machine_stats": {},
            "n_human": len(human_results),
            "n_machine": len(machine_results),
        }

    y_true = [1.0] * len(human_results) + [0.0] * len(machine_results)
    all_results = human_results + machine_results

    signal_extractors = {
        "entropy_production": lambda r: r.entropy_production,
        "integrated_information": lambda r: r.integrated_information,
        "temporal_binding": lambda r: r.temporal_binding,
        "free_energy_score": lambda r: r.free_energy_score,
        "causal_dag_mass": lambda r: r.signal_breakdown.get("raw", {}).get("causal_dag_mass", 0.0),
        "causal_concentration": lambda r: r.signal_breakdown.get("raw", {}).get("causal_concentration", 0.0),
        "cross_channel_mi": lambda r: r.signal_breakdown.get("raw", {}).get("cross_channel_mi", 0.0),
        "decay_type_score": lambda r: r.signal_breakdown.get("raw", {}).get("decay_type_score", 0.0),
        "adaptation_rate": lambda r: r.signal_breakdown.get("raw", {}).get("adaptation_rate", 0.0),
        "composite_consciousness_score": lambda r: r.composite_consciousness_score,
    }

    from .metrics import bootstrap_auc_ci

    per_metric_auc = {}
    per_metric_ci = {}
    for metric, extractor in signal_extractors.items():
        scores = [extractor(r) for r in all_results]
        point, ci_lo, ci_hi = bootstrap_auc_ci(y_true, scores, n_bootstrap=500)
        per_metric_auc[metric] = point
        per_metric_ci[metric] = (ci_lo, ci_hi)

    composite_auc = per_metric_auc.pop("composite_consciousness_score")
    composite_ci = per_metric_ci.pop("composite_consciousness_score")

    def _stats(results: List[ConsciousnessSignatureResult]) -> Dict[str, float]:
        composites = [r.composite_consciousness_score for r in results]
        n = len(composites)
        if n == 0:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        mean_val = sum(composites) / n
        return {
            "mean": round(mean_val, 4),
            "min": round(min(composites), 4),
            "max": round(max(composites), 4),
            "n_human_like": sum(1 for r in results if r.is_human_like),
        }

    result: Dict[str, Any] = {
        "per_metric_auc": per_metric_auc,
        "per_metric_ci": {k: {"ci_lower": v[0], "ci_upper": v[1]} for k, v in per_metric_ci.items()},
        "composite_auc": composite_auc,
        "composite_ci": {"ci_lower": composite_ci[0], "ci_upper": composite_ci[1]},
        "human_stats": _stats(human_results),
        "machine_stats": _stats(machine_results),
        "n_human": len(human_results),
        "n_machine": len(machine_results),
    }

    # Tiered evaluation: signal × tier AUC breakdown
    if machine_traces_by_tier:
        by_tier: Dict[str, Dict[str, Any]] = {}
        for tier_name, tier_traces in machine_traces_by_tier.items():
            tier_results = [compute_consciousness_signatures(t) for t in tier_traces]
            if not tier_results:
                continue
            tier_y = [1.0] * len(human_results) + [0.0] * len(tier_results)
            tier_all = human_results + tier_results
            tier_aucs: Dict[str, float] = {}
            for metric, extractor in signal_extractors.items():
                if metric == "composite_consciousness_score":
                    continue
                scores = [extractor(r) for r in tier_all]
                tier_aucs[metric] = round(compute_auc(tier_y, scores), 4)
            # Composite for this tier
            composite_scores = [r.composite_consciousness_score for r in tier_all]
            tier_composite = round(compute_auc(tier_y, composite_scores), 4)
            by_tier[tier_name] = {
                "per_signal_auc": tier_aucs,
                "composite_auc": tier_composite,
                "n_machine": len(tier_results),
                "machine_stats": _stats(tier_results),
            }
        result["by_tier"] = by_tier

    return result
