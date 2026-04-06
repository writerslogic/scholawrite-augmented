"""Cross-detector evaluation harness for the adversary hierarchy.

Provides a unified Detector protocol wrapping both internal detectors
(NCD, causal coupling, consciousness signatures) and external APIs
(GPTZero, Originality.ai). Each detector is evaluated against all
adversary tiers to populate the adversary hierarchy table.

Environment variables:
    GPTZERO_API_KEY: API key for GPTZero (https://app.gptzero.me/app/api)
    ORIGINALITY_API_KEY: API key for Originality.ai
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from .metrics import auc
from .schema import CausalEvent

logger = logging.getLogger(__name__)

__all__ = [
    "Detector",
    "DetectorResult",
    "NcdDetector",
    "CausalCouplingDetector",
    "ConsciousnessDetector",
    "GptZeroDetector",
    "OriginalityDetector",
    "DetectorHarness",
    "HierarchyLevel",
    "ADVERSARY_HIERARCHY",
]


# ── Adversary Hierarchy ──────────────────────────────────────────────────

@dataclass(frozen=True)
class HierarchyLevel:
    """Formal adversary sophistication level."""
    level: int
    name: str
    knowledge: str
    description: str


ADVERSARY_HIERARCHY = [
    HierarchyLevel(0, "uninformed", "none",
                   "Adversary has no knowledge of the detection method"),
    HierarchyLevel(1, "feature_aware", "feature_names",
                   "Adversary knows which features are used but not weights/thresholds"),
    HierarchyLevel(2, "model_aware", "full_model",
                   "Adversary knows the full detection model including weights"),
    HierarchyLevel(3, "process_aware", "raw_process_data",
                   "Adversary has access to raw process data (keystrokes, timing)"),
    HierarchyLevel(4, "supply_chain", "training_data",
                   "Adversary has access to training data and can perform adversarial training"),
]


# ── Detector Protocol ─────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    """Unified detection result from any detector."""
    score: float  # 0.0 = certainly AI, 1.0 = certainly human
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Detector(ABC):
    """Abstract base for all AI text detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this detector."""

    @property
    @abstractmethod
    def hierarchy_level_survived(self) -> int:
        """Highest adversary level this detector survives (theoretical)."""

    @property
    @abstractmethod
    def signal_type(self) -> str:
        """Category: statistical, perturbation, process_structural, product_level."""

    @abstractmethod
    def detect(self, text: str, **kwargs: Any) -> DetectorResult:
        """Score a text. Higher = more likely human."""

    def detect_batch(self, texts: List[str], **kwargs: Any) -> List[DetectorResult]:
        """Score multiple texts. Default: sequential. Override for batch APIs."""
        return [self.detect(t, **kwargs) for t in texts]


# ── Internal Detectors ────────────────────────────────────────────────────

class NcdDetector(Detector):
    """Normalized Compression Distance baseline."""

    def __init__(self, reference_text: str = ""):
        self._reference = reference_text

    @property
    def name(self) -> str:
        return "NCD"

    @property
    def hierarchy_level_survived(self) -> int:
        return 0  # Defeated by any informed adversary

    @property
    def signal_type(self) -> str:
        return "statistical"

    def detect(self, text: str, **kwargs: Any) -> DetectorResult:
        from .baselines import _compression_discontinuity
        ref = kwargs.get("reference_text", self._reference)
        if not ref:
            return DetectorResult(score=0.5, error="no_reference_text")
        ncd = _compression_discontinuity(ref, text)
        # Invert: high NCD = anomaly = more likely AI = lower "human" score
        return DetectorResult(score=1.0 - ncd, raw_response={"ncd": ncd})


class CausalCouplingDetector(Detector):
    """Causal coupling from embodied simulation traces."""

    @property
    def name(self) -> str:
        return "CausalCoupling"

    @property
    def hierarchy_level_survived(self) -> int:
        return 1

    @property
    def signal_type(self) -> str:
        return "process_structural"

    def detect(self, text: str, **kwargs: Any) -> DetectorResult:
        trace: Optional[List[CausalEvent]] = kwargs.get("trace")
        if not trace:
            return DetectorResult(score=0.5, error="no_trace_provided")
        from .metrics import compute_causal_signatures
        sigs = compute_causal_signatures(trace)
        coupling = abs(sigs.get("coupling", 0.0))
        return DetectorResult(
            score=min(1.0, coupling),
            raw_response=sigs,
        )


class ConsciousnessDetector(Detector):
    """Consciousness-correlate composite score from 6 signals."""

    @property
    def name(self) -> str:
        return "ConsciousnessSignatures"

    @property
    def hierarchy_level_survived(self) -> int:
        return 1  # Expert adversary (L2) defeats it — proven empirically

    @property
    def signal_type(self) -> str:
        return "process_structural"

    def detect(self, text: str, **kwargs: Any) -> DetectorResult:
        trace: Optional[List[CausalEvent]] = kwargs.get("trace")
        if not trace:
            return DetectorResult(score=0.5, error="no_trace_provided")
        from .consciousness_signatures import compute_consciousness_signatures
        result = compute_consciousness_signatures(trace)
        return DetectorResult(
            score=result.composite_consciousness_score,
            raw_response={
                "composite": result.composite_consciousness_score,
                "is_human_like": result.is_human_like,
                "signal_breakdown": result.signal_breakdown.get("normalized", {}),
            },
        )


# ── External API Detectors ────────────────────────────────────────────────

class GptZeroDetector(Detector):
    """GPTZero API wrapper (https://api.gptzero.me)."""

    API_URL = "https://api.gptzero.me/v2/predict/text"

    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 30):
        self._api_key = api_key if api_key is not None else os.environ.get("GPTZERO_API_KEY", "")
        self._rpm = requests_per_minute
        self._last_call = 0.0

    @property
    def name(self) -> str:
        return "GPTZero"

    @property
    def hierarchy_level_survived(self) -> int:
        return 0  # Product-level statistical detector

    @property
    def signal_type(self) -> str:
        return "product_level"

    def _throttle(self) -> None:
        interval = 60.0 / self._rpm
        elapsed = time.monotonic() - self._last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call = time.monotonic()

    def detect(self, text: str, **kwargs: Any) -> DetectorResult:
        if not self._api_key:
            return DetectorResult(score=0.5, error="GPTZERO_API_KEY not set")

        import httpx

        self._throttle()
        try:
            resp = httpx.post(
                self.API_URL,
                json={"document": text},
                headers={
                    "x-api-key": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("GPTZero API error: %s", e)
            return DetectorResult(score=0.5, error=str(e))

        # Extract document-level scores
        documents = data.get("documents", [{}])
        doc = documents[0] if documents else {}
        probs = doc.get("class_probabilities", {})

        # Score: probability that text is human-written
        human_prob = probs.get("human", 0.5)

        return DetectorResult(
            score=round(human_prob, 4),
            raw_response={
                "class_probabilities": probs,
                "predicted_class": doc.get("predicted_class", "unknown"),
                "completely_generated_prob": doc.get("completely_generated_prob", 0.0),
                "overall_burstiness": doc.get("overall_burstiness", 0.0),
            },
        )


class OriginalityDetector(Detector):
    """Originality.ai API wrapper (https://api.originality.ai)."""

    API_URL = "https://api.originality.ai/api/v1/scan/ai"

    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 30):
        self._api_key = api_key if api_key is not None else os.environ.get("ORIGINALITY_API_KEY", "")
        self._rpm = requests_per_minute
        self._last_call = 0.0

    @property
    def name(self) -> str:
        return "Originality.ai"

    @property
    def hierarchy_level_survived(self) -> int:
        return 0

    @property
    def signal_type(self) -> str:
        return "product_level"

    def _throttle(self) -> None:
        interval = 60.0 / self._rpm
        elapsed = time.monotonic() - self._last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call = time.monotonic()

    def detect(self, text: str, **kwargs: Any) -> DetectorResult:
        if not self._api_key:
            return DetectorResult(score=0.5, error="ORIGINALITY_API_KEY not set")

        import httpx

        self._throttle()
        try:
            resp = httpx.post(
                self.API_URL,
                json={"content": text},
                headers={
                    "X-OAI-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Originality.ai API error: %s", e)
            return DetectorResult(score=0.5, error=str(e))

        # Score: probability of being original (human-written)
        ai_score = data.get("score", {}).get("ai", 0.5)
        original_score = data.get("score", {}).get("original", 0.5)

        return DetectorResult(
            score=round(original_score, 4),
            raw_response={
                "ai_score": ai_score,
                "original_score": original_score,
                "public_link": data.get("public_link", ""),
            },
        )


# ── Harness: Cross-Detector Evaluation ────────────────────────────────────

@dataclass
class TierResult:
    """Detection results for a single detector against a single adversary tier."""
    detector_name: str
    tier: str
    auc_score: float
    n_human: int
    n_machine: int
    human_mean: float
    machine_mean: float
    errors: int = 0
    y_true: List[float] = field(default_factory=list)
    y_score: List[float] = field(default_factory=list)


class DetectorHarness:
    """Evaluate multiple detectors against multiple adversary tiers.

    Produces the adversary hierarchy table: detector × tier → AUC.
    """

    def __init__(self, detectors: Optional[List[Detector]] = None):
        self.detectors = detectors or []

    def add_detector(self, detector: Detector) -> None:
        self.detectors.append(detector)

    def evaluate_on_texts(
        self,
        human_texts: List[str],
        machine_texts_by_tier: Dict[str, List[str]],
        **detect_kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate all detectors on text-based inputs (for external APIs).

        Returns detector × tier AUC matrix.
        """
        results: Dict[str, Dict[str, TierResult]] = {}

        for detector in self.detectors:
            results[detector.name] = {}

            # Score human texts
            human_results = detector.detect_batch(human_texts, **detect_kwargs)
            human_scores = [r.score for r in human_results if r.error is None]
            human_errors = sum(1 for r in human_results if r.error is not None)

            for tier, machine_texts in machine_texts_by_tier.items():
                machine_results = detector.detect_batch(machine_texts, **detect_kwargs)
                machine_scores = [r.score for r in machine_results if r.error is None]
                machine_errors = sum(1 for r in machine_results if r.error is not None)

                if human_scores and machine_scores:
                    y_true = [1.0] * len(human_scores) + [0.0] * len(machine_scores)
                    y_score = human_scores + machine_scores
                    auc_val = round(auc(y_true, y_score), 4)
                else:
                    y_true = []
                    y_score = []
                    auc_val = 0.5

                results[detector.name][tier] = TierResult(
                    detector_name=detector.name,
                    tier=tier,
                    auc_score=auc_val,
                    n_human=len(human_scores),
                    n_machine=len(machine_scores),
                    human_mean=round(mean(human_scores), 4) if human_scores else 0.0,
                    machine_mean=round(mean(machine_scores), 4) if machine_scores else 0.0,
                    errors=human_errors + machine_errors,
                    y_true=y_true,
                    y_score=y_score,
                )

            logger.info("Evaluated %s: %s", detector.name, {
                t: r.auc_score for t, r in results[detector.name].items()
            })

        return self._format_results(results)

    def evaluate_on_traces(
        self,
        human_traces: List[List[CausalEvent]],
        machine_traces_by_tier: Dict[str, List[List[CausalEvent]]],
        human_texts: Optional[List[str]] = None,
        machine_texts_by_tier: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate all detectors on trace-based inputs.

        Internal detectors use traces; external detectors use texts.
        If texts are provided alongside traces, external detectors use them.
        """
        results: Dict[str, Dict[str, TierResult]] = {}

        for detector in self.detectors:
            results[detector.name] = {}
            is_trace_based = detector.signal_type in ("process_structural",)

            if is_trace_based:
                # Score human traces
                human_scores = []
                for trace in human_traces:
                    r = detector.detect("", trace=trace)
                    if r.error is None:
                        human_scores.append(r.score)

                for tier, traces in machine_traces_by_tier.items():
                    machine_scores = []
                    for trace in traces:
                        r = detector.detect("", trace=trace)
                        if r.error is None:
                            machine_scores.append(r.score)

                    if human_scores and machine_scores:
                        y_true = [1.0] * len(human_scores) + [0.0] * len(machine_scores)
                        y_score = human_scores + machine_scores
                        auc_val = round(auc(y_true, y_score), 4)
                    else:
                        y_true = []
                        y_score = []
                        auc_val = 0.5

                    results[detector.name][tier] = TierResult(
                        detector_name=detector.name,
                        tier=tier,
                        auc_score=auc_val,
                        n_human=len(human_scores),
                        n_machine=len(machine_scores),
                        human_mean=round(mean(human_scores), 4) if human_scores else 0.0,
                        machine_mean=round(mean(machine_scores), 4) if machine_scores else 0.0,
                        y_true=y_true,
                        y_score=y_score,
                    )
            elif human_texts and machine_texts_by_tier:
                # External detectors: use texts
                human_results = detector.detect_batch(human_texts)
                human_scores = [r.score for r in human_results if r.error is None]

                for tier, texts in machine_texts_by_tier.items():
                    machine_results = detector.detect_batch(texts)
                    machine_scores = [r.score for r in machine_results if r.error is None]

                    if human_scores and machine_scores:
                        y_true = [1.0] * len(human_scores) + [0.0] * len(machine_scores)
                        y_score = human_scores + machine_scores
                        auc_val = round(auc(y_true, y_score), 4)
                    else:
                        y_true = []
                        y_score = []
                        auc_val = 0.5

                    results[detector.name][tier] = TierResult(
                        detector_name=detector.name,
                        tier=tier,
                        auc_score=auc_val,
                        n_human=len(human_scores),
                        n_machine=len(machine_scores),
                        human_mean=round(mean(human_scores), 4) if human_scores else 0.0,
                        machine_mean=round(mean(machine_scores), 4) if machine_scores else 0.0,
                        errors=sum(1 for r in human_results + machine_results if r.error),
                        y_true=y_true,
                        y_score=y_score,
                    )
            else:
                logger.warning(
                    "Skipping %s: external detector requires texts", detector.name
                )

        return self._format_results(results)

    def _format_results(
        self, results: Dict[str, Dict[str, TierResult]]
    ) -> Dict[str, Any]:
        """Format results as detector × tier AUC matrix + metadata."""
        # Build matrix
        detector_names = list(results.keys())
        all_tiers = set()
        for tier_results in results.values():
            all_tiers.update(tier_results.keys())
        tiers = sorted(all_tiers)

        matrix: Dict[str, Dict[str, float]] = {}
        for det in detector_names:
            matrix[det] = {}
            for tier in tiers:
                if tier in results[det]:
                    matrix[det][tier] = results[det][tier].auc_score
                else:
                    matrix[det][tier] = -1.0  # not evaluated

        # Detector metadata
        detector_meta = {}
        for det in self.detectors:
            detector_meta[det.name] = {
                "signal_type": det.signal_type,
                "hierarchy_level_survived": det.hierarchy_level_survived,
                "hierarchy_name": ADVERSARY_HIERARCHY[det.hierarchy_level_survived].name
                if det.hierarchy_level_survived < len(ADVERSARY_HIERARCHY) else "unknown",
            }

        # Per-tier detail
        detail: Dict[str, Dict[str, Any]] = {}
        for det, tier_results in results.items():
            detail[det] = {}
            for tier, tr in tier_results.items():
                detail[det][tier] = {
                    "auc": tr.auc_score,
                    "n_human": tr.n_human,
                    "n_machine": tr.n_machine,
                    "human_mean": tr.human_mean,
                    "machine_mean": tr.machine_mean,
                    "errors": tr.errors,
                    "y_true": tr.y_true,
                    "y_score": tr.y_score,
                }

        return {
            "matrix": matrix,
            "tiers": tiers,
            "detectors": detector_meta,
            "detail": detail,
        }
