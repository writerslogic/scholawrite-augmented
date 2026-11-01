"""Embodied scholar simulation with metabolic resource tracking."""
from __future__ import annotations

import math
import random
import re
import hashlib
import statistics
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from .schema import ResourceAllocation, CognitiveState
from .config import SimulationConfig, get_academic_markers_flat, get_sim_config, load_config

__all__ = ["EmbodiedScholar", "erode_context_deterministically", "get_syntactic_demand", "get_embodied_state"]

_NORMAL = statistics.NormalDist()
_WRITER_PARAMS_CONFIG = "klicke_writer_params_train"
_LATENCY_FLOOR_MS = 50.0
_RHO_AR_MAX = 0.98


@dataclass(frozen=True)
class _WriterTiming:
    """One KLiCKe TRAIN writer: non-pause log-IKI moments and the pause mixture."""
    mu: float
    sigma: float
    rho: float
    p_pause: float
    pause_mu: float
    pause_sigma: float


@lru_cache(maxsize=1)
def _timing_writer_table() -> tuple[tuple[_WriterTiming, ...], tuple[float, ...], float, float]:
    """TRAIN writer rows, cumulative bootstrap weights, and the pause window [lo_ms, hi_ms).

    Built by analysis/calibration/build_writer_params.py. A row is drawn with probability
    proportional to its number of 50-IKI windows, the unit the calibration compares."""
    data = load_config(_WRITER_PARAMS_CONFIG)
    if not data:
        raise FileNotFoundError(
            f"timing_model='klicke_ar1' needs configs/{_WRITER_PARAMS_CONFIG}.json "
            "(analysis/calibration/build_writer_params.py)"
        )
    col = {name: i for i, name in enumerate(data["columns"])}
    rows, cumulative, total = [], [], 0.0
    for r in data["rows"]:
        rows.append(_WriterTiming(*(float(r[col[k]]) for k in _WriterTiming.__dataclass_fields__)))
        total += float(r[col["n_windows"]])
        cumulative.append(total)
    if not rows or total <= 0.0:
        raise ValueError(f"configs/{_WRITER_PARAMS_CONFIG}.json has no usable rows")
    meta = data["meta"]
    return tuple(rows), tuple(cumulative), float(meta["pause_threshold_ms"]), float(meta["iki_filter_ms"][1])


class _KlickeTiming:
    """State of timing_model='klicke_ar1' for one scholar.

    Randomness comes only from ``self.rng``, seeded from sha256(author_id, timing_seed), so the legacy
    ``EmbodiedScholar._rng`` stream is untouched. Four variates are drawn per event whatever the
    parameter values, which keeps common random numbers across a parameter sweep.
    """

    def __init__(self, author_id: str, cfg: SimulationConfig):
        self.cfg = cfg
        digest = hashlib.sha256(f"{author_id}:timing:{cfg.timing_seed}".encode()).hexdigest()
        self.rng = random.Random(int(digest, 16))
        rows, cumulative, lo_ms, hi_ms = _timing_writer_table()
        pick = bisect_right(cumulative, self.rng.random() * cumulative[-1])
        self.row = rows[min(pick, len(rows) - 1)]

        total_sd = cfg.timing_sigma_scale * self.row.sigma
        self._s_ar = total_sd * math.sqrt(cfg.timing_ar_share)
        self._s_wn = total_sd * math.sqrt(1.0 - cfg.timing_ar_share)
        self._rho = min(_RHO_AR_MAX, max(0.0, cfg.timing_rho_scale * self.row.rho))
        self._innovation_sd = math.sqrt(1.0 - self._rho ** 2)
        self._z: float | None = None  # unit-variance AR(1); starts from its stationary law

        # Pause durations: normal on log IKI truncated to [lo_ms, hi_ms), by inverse CDF.
        a = (math.log(lo_ms) - self.row.pause_mu) / self.row.pause_sigma
        b = (math.log(hi_ms) - self.row.pause_mu) / self.row.pause_sigma
        self._reflect = a > 0.0  # keep the probabilities in the small tail where doubles are accurate
        lo, hi = (_NORMAL.cdf(-b), _NORMAL.cdf(-a)) if self._reflect else (_NORMAL.cdf(a), _NORMAL.cdf(b))
        self._p_lo, self._p_span = lo, hi - lo
        self._hi_ms = hi_ms

    def _pause_ms(self, u: float) -> float:
        p = max(self._p_lo + u * self._p_span, 1e-300)
        z = -_NORMAL.inv_cdf(p) if self._reflect else _NORMAL.inv_cdf(p)
        return min(math.exp(self.row.pause_mu + self.row.pause_sigma * z), self._hi_ms - 0.01)

    def latency_ms(self, modulation: float) -> float:
        """One IKI. ``modulation`` = deterministic cognitive term / its value at depth 1, glucose 1."""
        rng, row, cfg = self.rng, self.row, self.cfg
        u_pause, eta, eps, u_len = rng.random(), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.random()
        if u_pause < row.p_pause:  # a pause leaves the AR state where it was
            ms = self._pause_ms(u_len)
            if cfg.timing_pause_modulated:
                ms *= modulation
        else:
            self._z = eta if self._z is None else self._rho * self._z + self._innovation_sd * eta
            log_base = row.mu + cfg.timing_offset + self._s_ar * self._z + self._s_wn * eps
            ms = math.exp(log_base) * modulation
        return round(max(_LATENCY_FLOOR_MS, ms), 2)


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
        self._timing = _KlickeTiming(author_id, self.config) if self.config.timing_model == "klicke_ar1" else None

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

    def deterministic_latency(self, syntactic_depth: float, glucose: float | None = None) -> float:
        """Noise-free cognitive latency term (ms) at the given depth and (default: current) glucose."""
        g = self.glucose if glucose is None else glucose
        return 115 + 90 * math.log(1 + syntactic_depth) * (1.12 - g)

    def calculate_latency(self, syntactic_depth: float) -> float:
        """Mapping from state to keystroke latency (ms).

        timing_model "lognormal_iid" (default): log-normal noise around the deterministic term, drawn
        from the scholar's seeded RNG. "klicke_ar1": KLiCKe-calibrated AR(1) + pause mixture whose
        level is scaled by the deterministic term relative to its value at depth 1, glucose 1.
        """
        if self._timing is not None:
            reference = self.deterministic_latency(1.0, 1.0)
            return self._timing.latency_ms(self.deterministic_latency(syntactic_depth) / reference)
        deterministic = self.deterministic_latency(syntactic_depth)
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
