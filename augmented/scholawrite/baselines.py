"""Baseline detection methods including NCD, causal coupling, and text-level analysis.

Provides both process-level baselines (NCD, causal coupling) and product-level
baselines (perplexity proxy, n-gram burstiness, stylometric divergence) for
comparison in the paper's detector hierarchy.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from .metrics import auc, f1, compute_causal_signatures, NCD_DEFAULT_THRESHOLD
from .schema import AugmentedDocument, InjectionSpan
from .text import get_token_count

__all__ = [
    "run_baselines",
    "detect_ncd_anomalies",
    "compute_detectability_score",
    "DetectabilityScore",
    "perplexity_proxy_score",
    "ngram_burstiness_score",
    "stylometric_divergence_score",
]

def run_baselines(docs: Sequence[AugmentedDocument], threshold: float = NCD_DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Execute baseline suite with per-injection-level and per-trajectory-state metrics.

    Returns overall AUC/F1 plus breakdowns by injection level and trajectory state,
    enabling fine-grained analysis of which conditions are hardest to detect.
    """
    y_true, y_ncd, y_coupling = [], [], []
    # Per-class tracking: injection_level and trajectory_state
    level_scores: Dict[str, Dict[str, list]] = {}
    state_scores: Dict[str, Dict[str, list]] = {}

    for doc in docs:
        prev_text = ""
        for rev in doc.revisions:
            has_inj = any(ann.label.is_injection() for ann in rev.annotations)
            y_true.append(1 if has_inj else 0)

            # 1. Statistical Baseline (NCD)
            score = _compression_discontinuity(prev_text, rev.text) if prev_text else 0.0
            y_ncd.append(score)

            # 2. Causal Baseline (Resource Coupling)
            coupling_score = 0.0
            for ann in rev.annotations:
                if ann.label.is_injection() and ann.causal_trace:
                    sigs = compute_causal_signatures(ann.causal_trace)
                    coupling_score = max(coupling_score, abs(sigs["coupling"]))
            y_coupling.append(coupling_score)

            # Track per-class for injections
            for ann in rev.annotations:
                if ann.label.is_injection():
                    if ann.injection_level:
                        lv = ann.injection_level.value
                        level_scores.setdefault(lv, {"y": [], "ncd": [], "coupling": []})
                        level_scores[lv]["y"].append(1)
                        level_scores[lv]["ncd"].append(score)
                        level_scores[lv]["coupling"].append(coupling_score)
                    if ann.trajectory_state:
                        st = ann.trajectory_state.value
                        state_scores.setdefault(st, {"y": [], "ncd": [], "coupling": []})
                        state_scores[st]["y"].append(1)
                        state_scores[st]["ncd"].append(score)
                        state_scores[st]["coupling"].append(coupling_score)

            prev_text = rev.text

    y_pred_ncd = [1 if s > threshold else 0 for s in y_ncd]
    tp = sum(1 for t, p in zip(y_true, y_pred_ncd) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred_ncd) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred_ncd) if t == 1 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    result: Dict[str, Any] = {
        "ncd_auc": round(auc(y_true, y_ncd), 4),
        "causal_coupling_auc": round(auc(y_true, y_coupling), 4),
        "overall_f1": round(f1(y_true, y_pred_ncd), 4),
        "overall_precision": round(precision, 4),
        "overall_recall": round(recall, 4),
    }

    # Per-injection-level metrics
    per_level: Dict[str, Dict[str, float]] = {}
    for lv, data in level_scores.items():
        n = len(data["y"])
        avg_ncd = sum(data["ncd"]) / n if n else 0.0
        avg_coupling = sum(data["coupling"]) / n if n else 0.0
        per_level[lv] = {
            "count": n,
            "mean_ncd": round(avg_ncd, 4),
            "mean_coupling": round(avg_coupling, 4),
            "detection_rate": round(sum(1 for s in data["ncd"] if s > threshold) / max(n, 1), 4),
        }
    result["per_injection_level"] = per_level

    # Per-trajectory-state metrics
    per_state: Dict[str, Dict[str, float]] = {}
    for st, data in state_scores.items():
        n = len(data["y"])
        avg_ncd = sum(data["ncd"]) / n if n else 0.0
        avg_coupling = sum(data["coupling"]) / n if n else 0.0
        per_state[st] = {
            "count": n,
            "mean_ncd": round(avg_ncd, 4),
            "mean_coupling": round(avg_coupling, 4),
            "detection_rate": round(sum(1 for s in data["ncd"] if s > threshold) / max(n, 1), 4),
        }
    result["per_trajectory_state"] = per_state

    return result

def detect_ncd_anomalies(prev_text: str, curr_text: str, threshold: float = NCD_DEFAULT_THRESHOLD) -> bool:
    """High-performance NCD-based anomaly detector.

    Uses Normalized Compression Distance to detect discontinuities
    between consecutive revisions that may indicate injection.

    Args:
        prev_text: Previous revision text
        curr_text: Current revision text
        threshold: NCD threshold (default: NCD_DEFAULT_THRESHOLD = 0.45)
                  Empirically tuned to balance precision/recall.

    Returns:
        True if NCD exceeds threshold, indicating potential anomaly.

    See docs/THRESHOLDS.md, Section "Detection Thresholds".
    """
    return _compression_discontinuity(prev_text, curr_text) > threshold

@dataclass(frozen=True)
class DetectabilityScore:
    """Per-injection detectability assessment across multiple baselines."""
    injection_id: str
    injection_level: Optional[str]
    ncd_discontinuity: float
    stylometric_divergence: float
    coupling_strength: float
    composite_score: float

    @property
    def difficulty_label(self) -> str:
        if self.composite_score < 0.3:
            return "hard"
        elif self.composite_score < 0.6:
            return "medium"
        return "easy"


def compute_detectability_score(
    span: InjectionSpan,
    preceding_text: str,
    span_text: str,
    following_text: str,
) -> DetectabilityScore:
    """Compute how detectable an injection is across multiple baselines.

    Returns a composite score where lower = harder to detect.
    """
    # 1. NCD discontinuity vs preceding text
    ncd = _compression_discontinuity(preceding_text, span_text) if preceding_text else 0.0

    # 2. Stylometric divergence (type-token ratio + sentence length variance)
    span_words = span_text.split()
    context_words = (preceding_text + " " + following_text).split()
    span_ttr = len(set(span_words)) / max(len(span_words), 1)
    ctx_ttr = len(set(context_words)) / max(len(context_words), 1)
    ttr_diff = abs(span_ttr - ctx_ttr)

    # Sentence length variance divergence
    span_sents = [s.strip() for s in re.split(r'[.!?]+', span_text) if s.strip()]
    ctx_sents = [s.strip() for s in re.split(r'[.!?]+', preceding_text + " " + following_text) if s.strip()]
    span_avg_len = sum(len(s.split()) for s in span_sents) / max(len(span_sents), 1)
    ctx_avg_len = sum(len(s.split()) for s in ctx_sents) / max(len(ctx_sents), 1)
    len_diff = abs(span_avg_len - ctx_avg_len) / max(ctx_avg_len, 1)

    stylometric = min(1.0, (ttr_diff + len_diff) / 2)

    # 3. Causal coupling strength
    coupling_strength = 0.0
    if span.causal_trace:
        sigs = compute_causal_signatures(span.causal_trace)
        coupling_strength = abs(sigs.get("coupling", 0.0))

    # Composite: higher = easier to detect
    composite = 0.4 * ncd + 0.3 * stylometric + 0.3 * (1.0 - min(1.0, coupling_strength))

    return DetectabilityScore(
        injection_id=span.injection_id,
        injection_level=span.injection_level.value if span.injection_level else None,
        ncd_discontinuity=round(ncd, 4),
        stylometric_divergence=round(stylometric, 4),
        coupling_strength=round(coupling_strength, 4),
        composite_score=round(composite, 4),
    )


def perplexity_proxy_score(text: str) -> float:
    """Estimate text perplexity via unigram log-probability.

    LLM-generated text tends to use higher-frequency tokens, yielding lower
    perplexity. Returns a score in [0, 1] where higher = more uniform (LLM-like).
    """
    words = text.lower().split()
    if len(words) < 5:
        return 0.5
    counts = Counter(words)
    total = len(words)
    vocab = len(counts)
    # Unigram entropy
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    # Max possible entropy = log2(vocab)
    max_entropy = math.log2(vocab) if vocab > 1 else 1.0
    # Normalized: lower entropy = more repetitive/predictable = LLM-like
    return round(min(1.0, entropy / max_entropy), 4)


def ngram_burstiness_score(text: str, n: int = 3) -> float:
    """Measure n-gram burstiness (variance in repetition patterns).

    Human text shows "bursty" n-gram usage (topic-specific clusters).
    LLM text distributes n-grams more uniformly. Returns variance of
    n-gram frequency distribution, normalized to [0, 1].
    """
    words = text.lower().split()
    if len(words) < n + 5:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    if not counts:
        return 0.0
    freqs = list(counts.values())
    mean_freq = sum(freqs) / len(freqs)
    variance = sum((f - mean_freq) ** 2 for f in freqs) / len(freqs)
    # Normalize by mean squared (coefficient of variation squared)
    if mean_freq < 1e-10:
        return 0.0
    cv_sq = variance / (mean_freq * mean_freq)
    return round(min(1.0, cv_sq), 4)


def stylometric_divergence_score(span_text: str, context_text: str) -> float:
    """Compute stylometric divergence between span and surrounding context.

    Measures: type-token ratio divergence, sentence length variance, and
    word length distribution divergence. Higher = more divergent = easier to detect.
    """
    span_words = span_text.split()
    ctx_words = context_text.split()
    if len(span_words) < 5 or len(ctx_words) < 5:
        return 0.0

    # 1. Type-token ratio divergence
    span_ttr = len(set(w.lower() for w in span_words)) / len(span_words)
    ctx_ttr = len(set(w.lower() for w in ctx_words)) / len(ctx_words)
    ttr_div = abs(span_ttr - ctx_ttr)

    # 2. Average word length divergence
    span_wl = sum(len(w) for w in span_words) / len(span_words)
    ctx_wl = sum(len(w) for w in ctx_words) / len(ctx_words)
    wl_div = abs(span_wl - ctx_wl) / max(ctx_wl, 1.0)

    # 3. Sentence length variance divergence
    span_sents = [s.strip() for s in re.split(r'[.!?]+', span_text) if s.strip()]
    ctx_sents = [s.strip() for s in re.split(r'[.!?]+', context_text) if s.strip()]
    span_sl = sum(len(s.split()) for s in span_sents) / max(len(span_sents), 1)
    ctx_sl = sum(len(s.split()) for s in ctx_sents) / max(len(ctx_sents), 1)
    sl_div = abs(span_sl - ctx_sl) / max(ctx_sl, 1.0)

    return round(min(1.0, (ttr_div + wl_div + sl_div) / 3.0), 4)


_ncd_cache: dict = {}

def _compression_discontinuity(a: str, b: str) -> float:
    """Normalized Compression Distance between two strings.

    Caches individual compressed lengths to avoid redundant compression calls.
    """
    def get_len(t: str) -> int:
        h = hash(t)
        if h not in _ncd_cache:
            _ncd_cache[h] = len(zlib.compress(t.encode("utf-8")))
        return _ncd_cache[h]
    ca, cb = get_len(a), get_len(b)
    cab = len(zlib.compress((a + " " + b).encode("utf-8")))
    if max(ca, cb) == 0: return 0.0
    return min(max((cab - min(ca, cb)) / max(ca, cb), 0.0), 1.0)
