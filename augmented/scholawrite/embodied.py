"""Embodied scholar simulation with metabolic resource tracking."""
from __future__ import annotations

import math
import random
import re
import hashlib
from functools import lru_cache
from typing import Optional
from .schema import ResourceAllocation, CognitiveState
from .config import get_academic_markers_flat, get_sim_config

__all__ = ["EmbodiedScholar", "erode_context_deterministically", "get_syntactic_demand", "get_embodied_state"]

class EmbodiedScholar:
    """
    A stateful agent that persists across an entire document's revision history.

    Simulates human cognitive resources including:
    - Glucose: Cognitive fuel that depletes irreversibly over time
    - Visual fatigue: Accumulated strain from screen exposure
    - Resource allocation: Dynamic distribution of cognitive capacity
    """
    def __init__(self, author_id: str, initial_glucose: Optional[float] = None):
        """Initialize the embodied scholar."""
        self.config = get_sim_config()
        self.author_id = author_id
        self.glucose = initial_glucose if initial_glucose is not None else self.config.initial_glucose
        self.visual_fatigue = 0.0
        self.total_tokens_produced = 0
        seed = int(hashlib.sha256(author_id.encode()).hexdigest()[:8], 16)
        self._rng = random.Random(seed)

    def consume_resources(self, tokens: int, syntactic_depth: float):
        """Irreversible metabolic depletion per token and complexity."""
        cfg = self.config
        # Complexity penalty: higher syntactic depth increases depletion
        complexity_penalty = 1.0 + (syntactic_depth / 10.0)

        # Logarithmic decay matching human metabolic study results
        self.glucose = max(cfg.glucose_floor, self.glucose * (cfg.glucose_depletion_rate ** (tokens * complexity_penalty)))

        # Visual fatigue accumulates with production
        self.visual_fatigue = min(1.0, self.visual_fatigue + (tokens / cfg.fatigue_divisor))
        self.total_tokens_produced += tokens

    def allocate_resources(self, demand: float) -> ResourceAllocation:
        """Deterministic allocation based on resource competition."""
        cfg = self.config
        # Lexical retrieval penalized by visual fatigue
        lexical = self.glucose * (1.0 - self.visual_fatigue * cfg.lexical_fatigue_penalty)
        # Syntactic planning with minimum floor
        syntactic = max(cfg.syntactic_min_floor, self.glucose * 1.3)
        # High demand triggers resource reallocation penalty
        if demand > cfg.high_syntactic_demand:
            syntactic *= 0.7
        # Attention with minimum floor
        attention = max(cfg.attention_min_floor, self.glucose - (self.visual_fatigue * cfg.attention_fatigue_penalty))

        return ResourceAllocation(
            lexical=round(min(1.0, lexical), 3),
            syntactic=round(min(1.0, syntactic), 3),
            attention=round(min(1.0, attention), 3)
        )

    def calculate_latency(self, syntactic_depth: float) -> float:
        """Mapping from state to keystroke latency (ms) with log-normal noise."""
        deterministic = 115 + 90 * math.log(1 + syntactic_depth) * (1.12 - self.glucose)
        sigma = self.config.latency_log_normal_sigma
        if sigma <= 0.0:
            return round(deterministic, 2)
        mu = math.log(max(deterministic, 1.0)) - sigma ** 2 / 2.0
        return round(max(50.0, self._rng.lognormvariate(mu, sigma)), 2)

    def get_biometric_salt(self, token_idx: int) -> str:
        """Generate a deterministic biometric salt for cryptographic anchoring.

        Uses only author_id and token_idx for cross-platform reproducibility.
        Glucose state is tracked separately in the causal trace.
        """
        return hashlib.sha256(f"{self.author_id}:{token_idx}".encode()).hexdigest()


def _estimate_minute(
    rev_idx: int,
    total_revs: int,
    timestamp: str | None = None,
    session_start: str | None = None,
) -> int:
    """Estimate session minute from timestamps or linear fallback."""
    if timestamp and session_start:
        try:
            from datetime import datetime
            fmt_candidates = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
            t_start = t_now = None
            for fmt in fmt_candidates:
                try:
                    t_start = datetime.strptime(session_start[:19], fmt)
                    t_now = datetime.strptime(timestamp[:19], fmt)
                    break
                except ValueError:
                    continue
            if t_start and t_now:
                delta_min = max(0, (t_now - t_start).total_seconds() / 60.0)
                return min(int(delta_min), 180)  # cap at 3 hours
        except Exception:
            pass
    # Fallback: linear interpolation over 90-minute session
    return int((rev_idx / max(1, total_revs)) * 90)


def get_embodied_state(
    author: EmbodiedScholar,
    rev_idx: int,
    total_revs: int,
    text_context: str = "",
    timestamp: str | None = None,
    session_start_timestamp: str | None = None,
) -> CognitiveState:
    """Sample the current cognitive state of the persistent author.

    Uses actual timestamps when both ``timestamp`` and ``session_start_timestamp``
    are provided; falls back to linear interpolation over a 90-minute session.
    """
    minute = _estimate_minute(rev_idx, total_revs, timestamp, session_start_timestamp)
    alloc = author.allocate_resources(get_syntactic_demand(text_context))
    return CognitiveState(
        minute=minute,
        fatigue_index=round(author.visual_fatigue, 3),
        glucose_level=round(author.glucose, 4),
        allocation=alloc,
        context_clarity=alloc.attention,
        biometric_salt=author.get_biometric_salt(rev_idx)
    )

_GENERIC_SUBSTITUTIONS = [
    "thing", "aspect", "element", "factor", "concept",
    "matter", "point", "issue", "area", "item",
]

# Function words to preserve during semantic erosion
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "as", "or", "and", "but",
    "if", "not", "no", "so", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "i", "you", "my", "his", "her", "our", "their",
})


def erode_context_deterministically(text: str, clarity: float, salt: str) -> str:
    """Simulate cognitive blurring by deterministically degrading context.

    At clarity < 0.9: orthographic erosion (punctuation loss, case errors).
    At clarity < 0.5: semantic erosion — 10-15% of content words replaced
    with generic substitutes, simulating lexical retrieval failure under
    fatigue (Flower & Hayes 1981).
    """
    if clarity >= 0.9 or not text: return text
    chars = list(text)
    threshold = 1.0 - clarity
    for i in range(len(chars)):
        h = int(hashlib.md5(f"{salt}:{i}".encode()).hexdigest(), 16) % 100
        if (h / 100.0) < threshold:
            if chars[i] in ",.!?;:": chars[i] = " "
            elif chars[i].isupper() and clarity < 0.6: chars[i] = chars[i].lower()
    result = "".join(chars)

    # Semantic erosion: replace content words with generic substitutes
    if clarity < 0.5:
        words = result.split()
        erosion_rate = min(0.15, (0.5 - clarity) * 0.3)  # 0-15% replacement
        new_words = []
        for wi, w in enumerate(words):
            if w.lower() in _FUNCTION_WORDS or len(w) <= 3:
                new_words.append(w)
                continue
            h = int(hashlib.md5(f"{salt}:w:{wi}".encode()).hexdigest(), 16) % 1000
            if (h / 1000.0) < erosion_rate:
                sub_idx = h % len(_GENERIC_SUBSTITUTIONS)
                new_words.append(_GENERIC_SUBSTITUTIONS[sub_idx])
            else:
                new_words.append(w)
        result = " ".join(new_words)

    return result

@lru_cache(maxsize=1)
def _get_marker_pattern() -> re.Pattern:
    """Build and cache the academic marker regex pattern from config."""
    markers = get_academic_markers_flat()
    if not markers:
        # Fallback to basic markers
        markers = [
            "however", "consequently", "whereas", "although", "notwithstanding",
            "furthermore", "moreover", "conversely", "nevertheless", "nonetheless",
        ]
    # Escape special regex characters and join with |
    escaped = [re.escape(m) for m in markers]
    pattern = r"(?i)\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern)


def get_syntactic_demand(text: str) -> float:
    """
    Heuristic for syntactic planning load calculation.

    Academic markers are loaded from configs/academic_markers.json,
    providing 100+ scholarly cohesive devices and technical connectors.
    """
    if not text:
        return 1.0
    words = text.split()

    # Use config-loaded markers for syntactic orchestration detection
    pattern = _get_marker_pattern()
    markers = len(pattern.findall(text))

    return max(1.0, min(10.0, (len(words) / 10.0) + (markers * 1.2)))
