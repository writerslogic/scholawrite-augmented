"""Token-granular causal execution engine for irreversible process simulation."""
from __future__ import annotations

import hashlib
import json
from statistics import mean, stdev
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from .embodied import EmbodiedScholar
from .config import get_discourse_markers, CONFIG_DIR, get_sim_config
from .schema import CausalEvent

__all__ = [
    "LexicalIntention",
    "ExecutionEvent",
    "DeterministicRepairGenerator",
    "IrreversibleProcessEngine",
]

@dataclass(frozen=True)
class LexicalIntention:
    """Token-granular intention with biometric grounding."""
    target: str
    syntactic_depth: float
    lexical_rarity: float
    cognitive_cost: float

@dataclass(frozen=True)
class ExecutionEvent:
    """Trace of a single token execution."""
    intention: LexicalIntention
    actual_output: str
    failure_mode: Optional[str]
    repair_distance: int
    glucose_before: float
    glucose_after: float
    latency_ms: float

class DeterministicRepairGenerator:
    """Generates repairs using a domain-specific exhaustive lexicon."""

    def __init__(self, discipline: str = "general_academic"):
        self.discipline = discipline
        self.lexicon = self._load_lexicon()

    def _load_lexicon(self) -> Dict[str, Tuple[str, ...]]:
        path = CONFIG_DIR / "lexical_substitutions.json"
        if not path.exists(): return {}
        try:
            full_data = json.loads(path.read_text(encoding="utf-8"))
            base = full_data.get("general_academic", {})
            specific = full_data.get(self.discipline.lower(), {})
            return {k.lower(): tuple(v) for k, v in {**base, **specific}.items()}
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return {}

    def generate_repair(self, intention: LexicalIntention, mode: str, glucose: float, token_idx: int) -> Tuple[str, int]:
        """Deterministic repair generation based on metabolic state."""
        seed = int(hashlib.md5(f"{intention.target}:{mode}:{glucose:.4f}:{token_idx}".encode()).hexdigest(), 16)

        # Load repair markers from config
        discourse_data = get_discourse_markers()
        repair_markers = discourse_data.get("repair_markers", {})

        if mode == "lexical_starvation":
            fallbacks = repair_markers.get("lexical_starvation_fallbacks", ["concept", "framework", "element"])
            subs = self.lexicon.get(intention.target.lower(), tuple(fallbacks))
            return f"{subs[seed % len(subs)]} ", 1
        if mode == "syntactic_collapse":
            markers = repair_markers.get("syntactic_collapse", ["thus, ", "so, ", "consequently, ", "accordingly, "])
            return markers[seed % len(markers)], 2

        # Cognitive overflow fallback
        overflow_fillers = repair_markers.get("cognitive_overflow_fillers", ["it "])
        return overflow_fillers[seed % len(overflow_fillers)], 3

class IrreversibleProcessEngine:
    """
    Simulates writing as an irreversible resource-coupled causal process.
    Operates directly on a stateful EmbodiedScholar.
    """

    def __init__(self, author: EmbodiedScholar, discipline: str = "general_academic"):
        self.config = get_sim_config()
        self.author = author
        self.token_idx = 0
        self.trace: List[ExecutionEvent] = []
        self.repair_gen = DeterministicRepairGenerator(discipline)

    def execute(self, intention: LexicalIntention) -> str:
        """Execute a single intention, permanently depleting the author's resources."""
        cfg = self.config
        glucose_before = self.author.glucose
        failure = self._check_failure(intention)

        if failure:
            output, repair_dist = self.repair_gen.generate_repair(intention, failure, self.author.glucose, self.token_idx)
            # Failure repair costs more
            depletion_cost = intention.cognitive_cost * cfg.failure_repair_cost_multiplier
        else:
            output = intention.target
            repair_dist = 0
            depletion_cost = intention.cognitive_cost

        latency = self.author.calculate_latency(intention.syntactic_depth)

        # Record event using state BEFORE permanent depletion
        self.trace.append(ExecutionEvent(
            intention=intention, actual_output=output,
            failure_mode=failure, repair_distance=repair_dist,
            glucose_before=glucose_before, glucose_after=max(cfg.glucose_floor, self.author.glucose - depletion_cost),
            latency_ms=round(latency, 2)
        ))

        # IRREVERSIBLE RESOURCE CONSUMPTION
        glucose_before_consume = self.author.glucose
        self.author.consume_resources(1, intention.syntactic_depth)
        self.author.glucose = max(cfg.glucose_floor, self.author.glucose - depletion_cost)
        if self.author.glucose > glucose_before_consume + 0.0001:
            raise ValueError(
                f"Irreversibility violated: glucose {glucose_before_consume:.6f} -> {self.author.glucose:.6f}"
            )

        self.token_idx += 1
        return output

    def _check_failure(self, intention: LexicalIntention) -> Optional[str]:
        """Deterministic resource-gated failure check."""
        cfg = self.config
        # Lexical starvation: low glucose + high lexical rarity
        if self.author.glucose < cfg.glucose_lexical_starvation and intention.lexical_rarity > (0.4 + (1.0 - self.author.glucose) * 0.6):
            return "lexical_starvation"
        # Syntactic collapse: depth exceeds glucose-scaled capacity
        if intention.syntactic_depth > (cfg.syntactic_collapse_base + self.author.glucose * cfg.syntactic_collapse_glucose_factor):
            return "syntactic_collapse"
        return None

    def compute_causal_signatures(self) -> Dict[str, Any]:
        """Mathematically rigorous biometric validation."""
        cfg = self.config
        if not self.trace:
            return {"repair_locality": 0.0, "resource_coupling": 0.0, "is_plausible": False, "causal_asymmetry": 0.0}

        fails = [i for i, e in enumerate(self.trace) if e.failure_mode]
        repairs = [i for i, e in enumerate(self.trace) if e.repair_distance > 0]

        # 1. Repair Locality
        locality = sum(min([abs(r-f) for r in repairs if r>=f] or [100]) for f in fails) / len(fails) if fails else 0.0

        # 2. Resource Coupling
        coupling = 0.0
        if len(self.trace) > cfg.min_trace_length_coupling and fails:
            try:
                x = [1 if e.failure_mode else 0 for e in self.trace[:-1]]
                y = [e.intention.syntactic_depth for e in self.trace[1:]]
                mu_x, mu_y = mean(x), mean(y)
                std_x, std_y = stdev(x), stdev(y)
                if std_x > 0 and std_y > 0:
                    coupling = sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / ((len(x)-1) * std_x * std_y)
            except (ValueError, ZeroDivisionError, TypeError):
                import logging
                logging.getLogger(__name__).warning("Coupling computation failed in causal engine for trace of length %d", len(self.trace))

        # Continuous plausibility score (0.0-1.0)
        locality_score = 0.0
        if cfg.locality_human_max > cfg.locality_human_min:
            if cfg.locality_human_min <= locality <= cfg.locality_human_max:
                midpoint = (cfg.locality_human_min + cfg.locality_human_max) / 2
                half_range = (cfg.locality_human_max - cfg.locality_human_min) / 2
                locality_score = 1.0 - abs(locality - midpoint) / half_range
        coupling_score = min(1.0, abs(coupling) / cfg.coupling_strong_threshold) if cfg.coupling_strong_threshold > 0 else 0.0
        plausibility = round(0.5 * locality_score + 0.5 * coupling_score, 3)

        # Granger causality asymmetry from the trace
        from .metrics import granger_causality_test
        causal_events = [
            CausalEvent(
                intention=e.intention.target,
                actual_output=e.actual_output,
                status="failure" if e.failure_mode else "success",
                failure_mode=e.failure_mode,
                repair_artifact=e.actual_output if e.repair_distance > 0 else None,
                glucose_at_event=e.glucose_before,
                latency_ms=e.latency_ms,
                syntactic_complexity=e.intention.syntactic_depth,
            )
            for e in self.trace
        ]
        causal_asymmetry = granger_causality_test(causal_events)

        return {
            "repair_locality": round(locality, 2),
            "resource_coupling": round(coupling, 3),
            "is_plausible": plausibility >= 0.5,
            "plausibility_score": plausibility,
            "causal_asymmetry": round(causal_asymmetry, 4),
        }

    def render_text(self) -> str:
        """Render text FROM causal trace."""
        return " ".join(e.actual_output for e in self.trace).strip()
