"""Causal signature primitives, classification metrics, and threshold constants.

Handles two distinct types of metrics:
1. Causal signature metrics for forensic validation (locality, coupling, plausibility)
2. Classification metrics for model evaluation (accuracy, F1, confusion matrix)

DOES NOT contain:
- Label parsing -> lives in labels.py
- Text manipulation -> lives in text.py
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, List, Tuple, Dict, Any, Optional
from .schema import CausalEvent
from .config import get_sim_config
from .thermodynamic import compute_entropy_production_rate
from .integrated_information import compute_phi
from .temporal_binding import compute_temporal_binding_index
from .free_energy import compute_free_energy_trajectory
from .consciousness_signatures import compute_consciousness_signatures as _compute_consciousness

logger = logging.getLogger(__name__)

# Edit Classification
SUBSTANTIAL_EDIT_RATIO = 0.20

# Detection Thresholds
NCD_DEFAULT_THRESHOLD = 0.45
NCD_HARDENING_DELTA = 0.10
JACCARD_LARGE_DIFF_THRESHOLD = 0.30
TRUNCATION_THRESHOLD = 0.40

# Trace Validation
MIN_TRACE_LENGTH_METRICS = 5
GLUCOSE_INCREASE_TOLERANCE = 0.0001

# Trajectory State Thresholds
COUPLING_ASSIMILATED_THRESHOLD = 0.7
COUPLING_WARM_MINIMUM = 0.4
LOCALITY_WARM_MAX = 4.0

# Anomaly Detection Thresholds (must be strictly above LOCALITY_WARM_MAX)
LOCALITY_ANOMALY_THRESHOLD = 4.5
COUPLING_ANOMALY_THRESHOLD = 0.2

__all__ = [
    # Metrics functions
    "auc", "f1", "span_iou", "compute_causal_signatures",
    "compute_classification_metrics", "granger_causality_test", "granger_causality_detailed", "bootstrap_auc_ci",
    "compute_fano_bound", "compute_conditional_auc",
    # Edit thresholds
    "SUBSTANTIAL_EDIT_RATIO",
    # Detection thresholds
    "NCD_DEFAULT_THRESHOLD", "NCD_HARDENING_DELTA",
    "JACCARD_LARGE_DIFF_THRESHOLD", "TRUNCATION_THRESHOLD",
    # Trace thresholds
    "MIN_TRACE_LENGTH_METRICS",
    "GLUCOSE_INCREASE_TOLERANCE",
    # Trajectory state thresholds
    "COUPLING_ASSIMILATED_THRESHOLD",
    "COUPLING_WARM_MINIMUM",
    "LOCALITY_WARM_MAX",
    # Anomaly detection thresholds
    "LOCALITY_ANOMALY_THRESHOLD",
    "COUPLING_ANOMALY_THRESHOLD",
]

def granger_causality_test(trace: List[CausalEvent]) -> float:
    """Compute causal asymmetry via lag-1 Granger causality F-test.

    Tests whether failure_flags[t] Granger-causes complexity[t+1] (forward)
    vs complexity[t] Granger-causing failure[t+1] (reverse). Returns
    F_forward / F_reverse as an asymmetry score. Human writing should show
    asymmetry > 1 (failures predict simplification); LLM text should be ~1.

    Uses OLS: regress y[t+1] on (y[t], x[t]) vs y[t+1] on (y[t]) alone.
    F = ((RSS_restricted - RSS_unrestricted) / 1) / (RSS_unrestricted / (n-3)).

    Assumptions: lag=1, stationarity not checked. Granger causality implies
    temporal precedence, not structural causality. Confounders (e.g., glucose
    driving both failure and complexity) may bias results.
    """
    if len(trace) < 4:
        return 0.0

    failure_flags = [1.0 if e.status != "success" else 0.0 for e in trace]
    complexities = [e.syntactic_complexity for e in trace]

    def _ols_f_stat(x: List[float], y: List[float]) -> float:
        """F-stat testing whether x[t] improves prediction of y[t+1] beyond y[t]."""
        n = len(x) - 1
        if n < 3:
            return 0.0

        # Restricted model: y[t+1] = a + b*y[t]
        # Unrestricted model: y[t+1] = a + b*y[t] + c*x[t]
        y_dep = y[1:]  # y[t+1]
        y_lag = y[:-1]  # y[t]
        x_lag = x[:-1]  # x[t]

        # OLS for restricted model: y_dep = a + b*y_lag
        n_obs = len(y_dep)
        sum_yl = sum(y_lag)
        sum_yd = sum(y_dep)
        sum_yl2 = sum(v * v for v in y_lag)
        sum_yl_yd = sum(a * b for a, b in zip(y_lag, y_dep))

        denom_r = n_obs * sum_yl2 - sum_yl * sum_yl
        if abs(denom_r) < 1e-15:
            return 0.0
        b_r = (n_obs * sum_yl_yd - sum_yl * sum_yd) / denom_r
        a_r = (sum_yd - b_r * sum_yl) / n_obs

        rss_restricted = sum((yd - a_r - b_r * yl) ** 2 for yd, yl in zip(y_dep, y_lag))

        # OLS for unrestricted model: y_dep = a + b*y_lag + c*x_lag
        # Normal equations via 3x3 system: X'X beta = X'y
        # columns: [1, y_lag, x_lag]
        s1 = float(n_obs)
        sy = sum_yl
        sx = sum(x_lag)
        syy = sum_yl2
        sxx = sum(v * v for v in x_lag)
        sxy = sum(a * b for a, b in zip(y_lag, x_lag))
        sd = sum_yd
        sdy = sum_yl_yd
        sdx = sum(a * b for a, b in zip(y_dep, x_lag))

        # Solve X'X beta = X'y using Cramer's rule
        # X'X = [[s1, sy, sx], [sy, syy, sxy], [sx, sxy, sxx]]
        # X'y = [sd, sdy, sdx]
        det = (s1 * (syy * sxx - sxy * sxy)
               - sy * (sy * sxx - sxy * sx)
               + sx * (sy * sxy - syy * sx))
        if abs(det) < 1e-15:
            return 0.0

        a_u = ((sd * (syy * sxx - sxy * sxy)
                - sy * (sdy * sxx - sdx * sxy)
                + sx * (sdy * sxy - sdx * syy)) / det)
        b_u = ((s1 * (sdy * sxx - sdx * sxy)
                - sd * (sy * sxx - sxy * sx)
                + sx * (sy * sdx - sdy * sx)) / det)
        c_u = ((s1 * (syy * sdx - sxy * sdy)
                - sy * (sy * sdx - sdy * sx)
                + sd * (sy * sxy - syy * sx)) / det)

        rss_unrestricted = sum(
            (yd - a_u - b_u * yl - c_u * xl) ** 2
            for yd, yl, xl in zip(y_dep, y_lag, x_lag)
        )

        if rss_unrestricted < 1e-15:
            return 0.0

        # F = ((RSS_r - RSS_u) / 1) / (RSS_u / (n_obs - 3))
        df_denom = n_obs - 3
        if df_denom <= 0:
            return 0.0
        f_stat = ((rss_restricted - rss_unrestricted) / 1.0) / (rss_unrestricted / df_denom)
        return max(0.0, f_stat)

    # Forward: do failures Granger-cause complexity changes?
    f_forward = _ols_f_stat(failure_flags, complexities)
    # Reverse: does complexity Granger-cause failure changes?
    f_reverse = _ols_f_stat(complexities, failure_flags)

    if f_reverse < 1e-10:
        return f_forward  # avoid division by zero; pure forward causality
    return f_forward / f_reverse


def _f_survival(f_stat: float, df1: int, df2: int) -> float:
    """Approximate p-value for F(df1, df2) using regularized incomplete beta.

    Uses the relation: P(F > f) = I_x(df2/2, df1/2) where x = df2/(df2 + df1*f).
    Approximates via continued fraction expansion of the incomplete beta function.
    """
    if f_stat <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f_stat)
    a, b = df2 / 2.0, df1 / 2.0
    # Use log-gamma via Stirling for beta function normalization
    # For our use case (df1=1, df2~n-3), a simple series expansion suffices
    # Fall back to normal approximation for large df2
    if df2 > 100:
        # Approximation: F ~ chi2(1)/1 for df1=1, large df2
        # P(chi2(1) > f) ≈ 2*(1 - Phi(sqrt(f)))
        z = math.sqrt(f_stat)
        # Abramowitz & Stegun approximation for erfc
        t = 1.0 / (1.0 + 0.3275911 * z)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        p = poly * math.exp(-z * z / 2.0) * 0.3989422804  # 1/sqrt(2pi)
        return min(1.0, max(0.0, 2.0 * p))
    # For small df2, use series expansion of incomplete beta
    # I_x(a,b) via continued fraction (Lentz's method)
    prefix = math.exp(
        a * math.log(x) + b * math.log(1 - x)
        + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    ) / a
    # Simple 20-term continued fraction
    cf = 1.0
    for m in range(20, 0, -1):
        # Even term
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        cf = 1.0 + num / cf if abs(cf) > 1e-30 else 1.0 + num
        # Odd term
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        cf = 1.0 + num / cf if abs(cf) > 1e-30 else 1.0 + num
    return min(1.0, max(0.0, prefix / cf))


def granger_causality_detailed(trace: List[CausalEvent]) -> Dict[str, float]:
    """Granger causality with F-stats, p-values, and asymmetry ratio.

    Returns dict with f_forward, f_reverse, p_forward, p_reverse, asymmetry_ratio.
    """
    if len(trace) < 4:
        return {"f_forward": 0.0, "f_reverse": 0.0, "p_forward": 1.0, "p_reverse": 1.0, "asymmetry_ratio": 0.0}

    failure_flags = [1.0 if e.status != "success" else 0.0 for e in trace]
    complexities = [e.syntactic_complexity for e in trace]
    n_obs = len(trace) - 1

    # Reuse the existing function's inner logic
    asymmetry = granger_causality_test(trace)

    # Compute individual F-stats by calling the inner function pattern
    # We need to extract them separately
    def _ols_f_stat(x: List[float], y: List[float]) -> float:
        n = len(x) - 1
        if n < 3:
            return 0.0
        y_dep = y[1:]
        y_lag = y[:-1]
        x_lag = x[:-1]
        n_obs_inner = len(y_dep)
        sum_yl = sum(y_lag)
        sum_yd = sum(y_dep)
        sum_yl2 = sum(v * v for v in y_lag)
        sum_yl_yd = sum(a * b for a, b in zip(y_lag, y_dep))
        denom_r = n_obs_inner * sum_yl2 - sum_yl * sum_yl
        if abs(denom_r) < 1e-15:
            return 0.0
        b_r = (n_obs_inner * sum_yl_yd - sum_yl * sum_yd) / denom_r
        a_r = (sum_yd - b_r * sum_yl) / n_obs_inner
        rss_restricted = sum((yd - a_r - b_r * yl) ** 2 for yd, yl in zip(y_dep, y_lag))
        s1 = float(n_obs_inner)
        sy = sum_yl
        sx = sum(x_lag)
        syy = sum_yl2
        sxx = sum(v * v for v in x_lag)
        sxy = sum(a * b for a, b in zip(y_lag, x_lag))
        sd = sum_yd
        sdy = sum_yl_yd
        sdx = sum(a * b for a, b in zip(y_dep, x_lag))
        det = (s1 * (syy * sxx - sxy * sxy) - sy * (sy * sxx - sxy * sx) + sx * (sy * sxy - syy * sx))
        if abs(det) < 1e-15:
            return 0.0
        a_u = (sd * (syy * sxx - sxy * sxy) - sy * (sdy * sxx - sdx * sxy) + sx * (sdy * sxy - sdx * syy)) / det
        b_u = (s1 * (sdy * sxx - sdx * sxy) - sd * (sy * sxx - sxy * sx) + sx * (sy * sdx - sdy * sx)) / det
        c_u = (s1 * (syy * sdx - sxy * sdy) - sy * (sy * sdx - sdy * sx) + sd * (sy * sxy - syy * sx)) / det
        rss_unrestricted = sum((yd - a_u - b_u * yl - c_u * xl) ** 2 for yd, yl, xl in zip(y_dep, y_lag, x_lag))
        if rss_unrestricted < 1e-15:
            return 0.0
        df_denom = n_obs_inner - 3
        if df_denom <= 0:
            return 0.0
        return max(0.0, ((rss_restricted - rss_unrestricted) / 1.0) / (rss_unrestricted / df_denom))

    f_fwd = _ols_f_stat(failure_flags, complexities)
    f_rev = _ols_f_stat(complexities, failure_flags)
    df2 = n_obs - 3

    p_fwd = _f_survival(f_fwd, 1, df2) if df2 > 0 else 1.0
    p_rev = _f_survival(f_rev, 1, df2) if df2 > 0 else 1.0

    return {
        "f_forward": round(f_fwd, 4),
        "f_reverse": round(f_rev, 4),
        "p_forward": round(p_fwd, 6),
        "p_reverse": round(p_rev, 6),
        "asymmetry_ratio": round(asymmetry, 4),
    }


def compute_causal_signatures(trace: List[CausalEvent]) -> Dict[str, float]:
    """Compute repair locality, resource coupling, and plausibility from a causal trace."""
    cfg = get_sim_config()
    if not trace:
        return {"locality": 0.0, "coupling": 0.0, "plausibility": 0.0, "causal_asymmetry": 0.0, "entropy_production": 0.0, "integrated_information": 0.0, "temporal_binding": 0.0, "free_energy_score": 0.0, "consciousness_score": 0.0}

    failure_indices = [i for i, e in enumerate(trace) if e.status != "success"]
    repair_indices = [i for i, e in enumerate(trace) if e.repair_artifact]

    locality = 0.0
    if failure_indices and repair_indices:
        distances = []
        for f in failure_indices:
            next_repairs = [r for r in repair_indices if r >= f]
            if next_repairs:
                distances.append(min(next_repairs) - f + 1)
        locality = sum(distances) / len(distances) if distances else 0.0

    failure_flags = [1 if e.status != "success" else 0 for e in trace]
    complexities = [e.syntactic_complexity for e in trace]

    coupling = 0.0
    coupling_valid = False
    failure_count = sum(failure_flags)
    if len(trace) > MIN_TRACE_LENGTH_METRICS and failure_count >= 3:
        try:
            # Point-biserial correlation (Pearson on binary×continuous)
            from statistics import mean, stdev
            x = failure_flags[:-1]
            y = complexities[1:]
            mu_x, mu_y = mean(x), mean(y)
            std_x, std_y = stdev(x), stdev(y)
            if std_x > 0 and std_y > 0:
                coupling = sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / ((len(x) - 1) * std_x * std_y)
                coupling_valid = True
        except (ValueError, TypeError, ZeroDivisionError):
            logger.warning("Coupling computation failed for trace of length %d", len(trace))
            coupling = 0.0

    # Continuous plausibility score (0.0-1.0) instead of binary
    locality_score = 0.0
    if cfg.locality_human_max > cfg.locality_human_min:
        if cfg.locality_human_min <= locality <= cfg.locality_human_max:
            # Peak at midpoint of human range
            midpoint = (cfg.locality_human_min + cfg.locality_human_max) / 2
            half_range = (cfg.locality_human_max - cfg.locality_human_min) / 2
            locality_score = 1.0 - abs(locality - midpoint) / half_range
        else:
            locality_score = 0.0
    coupling_score = min(1.0, abs(coupling) / cfg.coupling_strong_threshold) if cfg.coupling_strong_threshold > 0 else 0.0
    plausibility = round(0.5 * locality_score + 0.5 * coupling_score, 3)

    causal_asymmetry = granger_causality_test(trace)

    entropy_production = compute_entropy_production_rate(trace)

    integrated_information = compute_phi(trace)

    temporal_binding = compute_temporal_binding_index(trace)

    free_energy = compute_free_energy_trajectory(trace)
    free_energy_score = free_energy["trajectory_score"]

    # Consciousness composite: only compute for traces long enough to be meaningful
    consciousness_score = 0.0
    if len(trace) > 15:
        cs_result = _compute_consciousness(trace)
        consciousness_score = cs_result.composite_consciousness_score

    return {
        "locality": round(locality, 2),
        "coupling": round(coupling, 3),
        "coupling_valid": coupling_valid,
        "plausibility": plausibility,
        "is_plausible": plausibility >= 0.5,
        "causal_asymmetry": round(causal_asymmetry, 4),
        "entropy_production": round(entropy_production, 6),
        "integrated_information": integrated_information,
        "temporal_binding": round(temporal_binding, 6),
        "free_energy_score": round(free_energy_score, 6),
        "consciousness_score": round(consciousness_score, 4),
    }

def auc(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    pairs = list(zip(y_score, y_true))
    if not pairs: return 0.0
    pos = sum(1 for _, y in pairs if y > 0)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0: return 0.0
    pairs.sort(key=lambda x: x[0])
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j): ranks[k] = avg_rank
        i = j
    sum_ranks_pos = sum(rank for rank, (_, y) in zip(ranks, pairs) if y > 0)
    return (sum_ranks_pos - (pos * (pos + 1)) / 2.0) / (pos * neg)

def bootstrap_auc_ci(
    y_true: List[float], y_score: List[float],
    n_bootstrap: int = 1000, ci_level: float = 0.95, seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute AUC with bootstrap confidence interval.

    Returns (point_auc, ci_lower, ci_upper).
    """
    import random as _rng
    point = auc(y_true, y_score)
    n = len(y_true)
    if n < 4:
        return (point, point, point)
    rng = _rng.Random(seed)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        bt = [y_true[i] for i in idx]
        bs = [y_score[i] for i in idx]
        if len(set(bt)) < 2:
            continue
        boot_aucs.append(auc(bt, bs))
    if not boot_aucs:
        return (point, point, point)
    boot_aucs.sort()
    alpha = (1 - ci_level) / 2
    lo = boot_aucs[max(0, int(alpha * len(boot_aucs)))]
    hi = boot_aucs[min(len(boot_aucs) - 1, int((1 - alpha) * len(boot_aucs)))]
    return (round(point, 4), round(lo, 4), round(hi, 4))


def f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute F1 score."""
    tp, fp, fn = 0, 0, 0
    for yt, yp in zip(y_true, y_pred):
        if yp and yt: tp += 1
        elif yp and not yt: fp += 1
        elif yt and not yp: fn += 1
    denom = (2 * tp + fp + fn)
    return (2 * tp) / denom if denom > 0 else 0.0

def span_iou(pred_spans: Iterable[Tuple[int, int]], true_spans: Iterable[Tuple[int, int]]) -> float:
    """Compute Intersection over Union for character spans."""
    def merge(sps):
        sps = sorted([(s, e) for s, e in sps if e > s])
        if not sps: return []
        res = [sps[0]]
        for s, e in sps[1:]:
            ls, le = res[-1]
            if s <= le: res[-1] = (ls, max(le, e))
            else: res.append((s, e))
        return res
    mp, mt = merge(pred_spans), merge(true_spans)
    if not mp and not mt: return 1.0
    if not mp or not mt: return 0.0
    inter, i, j = 0, 0, 0
    while i < len(mp) and j < len(mt):
        s, e = max(mp[i][0], mt[j][0]), min(mp[i][1], mt[j][1])
        if e > s: inter += e - s
        if mp[i][1] <= mt[j][1]: i += 1
        else: j += 1
    union = sum(e-s for s,e in mp) + sum(e-s for s,e in mt) - inter
    return inter / union if union > 0 else 0.0


def compute_classification_metrics(
    y_true: List[Any],
    y_pred: List[Any],
    labels: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute classification metrics for model evaluation.

    Consolidates the scattered calculate_metrics functions from:
    - analysis/classification_stats.py
    - scholawrite_finetune/bert_finetune/small_model_inference.py
    - scholawrite_finetune/bert_finetune/small_model_classifier.py
    - scholawrite_finetune/bert_finetune/small_model_analysis.py

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Optional list of label names for ordering. If None, derived from y_true.
        verbose: If True, print metrics to stdout. Defaults to True.

    Returns:
        Dict containing accuracy, macro_f1, micro_f1, and per-class metrics.

    Note:
        For confusion matrix visualization, use sklearn directly with matplotlib.
        This function focuses on computing numeric metrics without side effects.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "per_class": {}}

    # Compute accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / len(y_true)

    # Get all unique labels
    all_labels = labels or sorted(set(y_true) | set(y_pred))

    # Compute per-class precision, recall, F1
    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []

    for label in all_labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[str(label)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1_score, 4),
            "support": sum(1 for yt in y_true if yt == label),
        }
        f1_scores.append(f1_score)

    # Macro F1: average of per-class F1 scores
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Micro F1: same as accuracy for multiclass single-label classification
    micro_f1 = accuracy

    if verbose:
        print(f"accuracy: {accuracy:.4f}")
        print(f"macro_f1: {macro_f1:.4f}")
        print(f"micro_f1: {micro_f1:.4f}")

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class,
    }


def compute_fano_bound(
    error_rate: float,
    n_classes: int = 2,
) -> float:
    """Derive minimum mutual information needed via Fano's inequality.

    Fano's inequality: H(X|Y) <= h(Pe) + Pe * log2(|X|-1)
    where Pe = error probability, h(Pe) = binary entropy.

    For reliable classification (Pe < threshold), we need:
    I(X;Y) >= H(X) - H(X|Y) >= log2(n_classes) - h(Pe) - Pe * log2(n_classes - 1)

    Returns the minimum mutual information (bits) required.
    """
    if error_rate <= 0:
        return math.log2(n_classes)
    if error_rate >= 1.0:
        return 0.0

    # Binary entropy h(Pe)
    h_pe = -error_rate * math.log2(error_rate) - (1 - error_rate) * math.log2(1 - error_rate)

    # Fano upper bound on H(X|Y)
    h_x_given_y = h_pe + error_rate * math.log2(max(n_classes - 1, 1))

    # H(X) for uniform prior
    h_x = math.log2(n_classes)

    # Minimum MI needed
    mi_min = max(0.0, h_x - h_x_given_y)
    return round(mi_min, 6)


def compute_conditional_auc(
    y_true: List[float],
    signal_scores: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Compute conditional AUC: AUC of each signal given all others.

    For each signal, computes:
    - standalone AUC (signal alone)
    - marginal contribution (full - leave_one_out)
    - conditional mutual information proxy via AUC difference

    Returns dict with per-signal analysis and Fano bound estimate.
    """
    signal_names = list(signal_scores.keys())
    n = len(y_true)

    # Full composite AUC
    full_scores = [
        sum(signal_scores[s][i] for s in signal_names) / len(signal_names)
        for i in range(n)
    ]
    full_auc = auc(y_true, full_scores)

    # Error rate estimate from AUC (Pe ≈ 1 - AUC for balanced classes)
    pe_full = max(0.0, 1.0 - full_auc)
    fano_mi_needed = compute_fano_bound(pe_full)

    result: Dict[str, Any] = {
        "full_auc": round(full_auc, 4),
        "error_rate": round(pe_full, 4),
        "fano_mi_bound": round(fano_mi_needed, 6),
        "signals": {},
    }

    for sig in signal_names:
        # Standalone
        standalone_auc = auc(y_true, signal_scores[sig])

        # Leave-one-out
        remaining = [s for s in signal_names if s != sig]
        loo_scores = [
            sum(signal_scores[s][i] for s in remaining) / len(remaining)
            for i in range(n)
        ]
        loo_auc = auc(y_true, loo_scores)

        marginal = full_auc - loo_auc
        pe_standalone = max(0.0, 1.0 - standalone_auc)
        fano_standalone = compute_fano_bound(pe_standalone)

        result["signals"][sig] = {
            "standalone_auc": round(standalone_auc, 4),
            "loo_auc": round(loo_auc, 4),
            "marginal_contribution": round(marginal, 4),
            "standalone_error_rate": round(pe_standalone, 4),
            "fano_mi_needed": round(fano_standalone, 6),
        }

    return result
