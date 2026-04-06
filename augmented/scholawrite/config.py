"""Configuration loading utilities for ScholaWrite."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

__all__ = [
    "load_config",
    "get_leakage_patterns",
    "get_academic_markers",
    "get_sensory_anchors",
    "get_placeholder_text",
    "get_discourse_markers",
    "SimulationConfig",
    "get_sim_config",
    "CONFIG_DIR",
    "validate_configs",
    "snapshot_config",
]

@dataclass(frozen=True)
class SimulationConfig:
    """Consolidated thresholds and parameters for cognitive simulation.

    Parameter sources and refresh schedule:

    Metabolics:
        glucose_depletion_rate: Kellogg (1987) fatigue onset at 45-60 min.
            0.9988 yields glucose=0.45 at ~800 tokens. Review if new
            keystroke-fatigue studies emerge.
        fatigue_divisor: Calibrated to match Levy et al. (2013) pause
            distributions in academic writing.

    Resource Allocation:
        high_syntactic_demand: Aligned with Gibson (2000) dependency locality
            theory — syntactic depth >5 incurs disproportionate processing cost.
        lexical/attention penalties: Derived from Just & Carpenter (1992)
            capacity theory of comprehension.

    Failure Modes:
        glucose_lexical_starvation: 0.65 threshold from Flower & Hayes (1981)
            cognitive process model — writers begin lexical substitution at
            ~35% resource depletion.
        failure_repair_cost_multiplier: 2.7x based on Chenoweth & Hayes (2001)
            revision cost measurements.

    Next review: Update when keystroke-logging corpora (e.g., Leijten & Van Waes
    2013 Inputlog studies) provide new baseline measurements.
    """
    # Metabolics (EmbodiedScholar)
    initial_glucose: float = 1.0
    glucose_floor: float = 0.05
    glucose_depletion_rate: float = 0.9988
    fatigue_divisor: float = 12000.0

    # Resource Allocation
    high_syntactic_demand: float = 5.0
    lexical_fatigue_penalty: float = 0.4
    syntactic_min_floor: float = 0.3
    attention_min_floor: float = 0.1
    attention_fatigue_penalty: float = 0.5

    # Failure Mode Thresholds
    glucose_lexical_starvation: float = 0.65
    syntactic_collapse_base: float = 4.0
    syntactic_collapse_glucose_factor: float = 3.5
    failure_repair_cost_multiplier: float = 2.7

    # LLM Generation Mapping
    temp_min: float = 0.5
    temp_max: float = 0.9
    top_p_min: float = 0.8
    top_p_max: float = 0.98
    presence_penalty_max: float = 0.5
    freq_penalty_max: float = 0.3
    max_tokens_base: int = 1536
    max_tokens_scale: int = 1536

    # Keystroke Latency Noise
    # Log-normal sigma for IKI variance. 0.9 targets CV~1.5, matching
    # KLiCKe composition writers (empirical sigma=1.57, conservative subset).
    # Set to 0.0 for fully deterministic traces.
    latency_log_normal_sigma: float = 0.9

    # Forensic Signatures
    locality_human_min: float = 1.0
    locality_human_max: float = 3.5
    coupling_strong_threshold: float = 0.6
    min_trace_length_coupling: int = 10

@lru_cache(maxsize=1)
def get_sim_config() -> SimulationConfig:
    """Load simulation configuration from disk or return defaults."""
    data = load_config("simulation_config")
    if not data:
        import logging
        logging.getLogger(__name__).warning(
            "simulation_config.json not found in %s; using default parameters", CONFIG_DIR
        )
        return SimulationConfig()
    # Filter out non-field keys (e.g., _comment)
    valid_fields = {f.name for f in SimulationConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return SimulationConfig(**filtered)


def snapshot_config(config: Optional[SimulationConfig] = None) -> dict:
    """Serialize current simulation config for audit trail / reproducibility.

    Returns a dict with all parameter values, suitable for embedding in
    RunManifest or experiment result JSON files.
    """
    from dataclasses import asdict
    cfg = config or get_sim_config()
    return asdict(cfg)


# Default config directory - can be overridden
# Path: augmented/scholawrite/config.py -> augmented/configs/
CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a JSON config file by name.

    Args:
        name: Config file name (without .json extension)
        config_dir: Optional custom config directory

    Returns:
        Parsed JSON as a dictionary, or empty dict if not found
    """
    base_dir = config_dir or CONFIG_DIR
    path = base_dir / f"{name}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        import logging
        logging.getLogger(__name__).warning("Failed to load config %s: %s", name, e)
        return {}


@lru_cache(maxsize=1)
def get_leakage_patterns() -> List[str]:
    """Load leakage detection patterns from config.

    Returns:
        List of regex pattern strings with case-insensitivity where appropriate
    """
    data = load_config("leakage_patterns")
    if not data:
        # Fallback minimal patterns
        return [
            r"(?i)\bas an ai\b",
            r"(?i)^here is\b",
            r"(?i)^certainly!",
        ]

    patterns = []
    for category, pattern_list in data.items():
        if category.startswith("_"):
            continue
        if isinstance(pattern_list, list):
            for p in pattern_list:
                # Add case-insensitivity flag if not already present
                # Handle patterns that start with ^ (anchored)
                if p.startswith("(?i)"):
                    patterns.append(p)
                elif p.startswith("^"):
                    # Anchor pattern - insert (?i) after ^
                    patterns.append(f"(?i){p}")
                else:
                    # Regular pattern
                    patterns.append(f"(?i){p}")
    return patterns


@lru_cache(maxsize=1)
def get_academic_markers() -> Dict[str, Any]:
    """Load academic markers from config.

    Returns:
        Dictionary with marker categories and lists
    """
    data = load_config("academic_markers")
    if not data:
        # Fallback minimal markers
        return {
            "logical_connectors": {
                "contrast": ["however", "nevertheless"],
                "causation": ["consequently", "therefore"],
            }
        }
    return data


def get_academic_markers_flat() -> List[str]:
    """Get a flat list of all academic markers for regex matching.

    Returns:
        List of all marker words/phrases
    """
    data = get_academic_markers()
    markers = []

    def extract_strings(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    markers.append(item)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key.startswith("_"):
                    continue
                extract_strings(value)

    extract_strings(data)
    return markers


@lru_cache(maxsize=1)
def get_sensory_anchors() -> Dict[str, Any]:
    """Load sensory anchor phrases from config.

    Returns:
        Dictionary with anchors and weights
    """
    data = load_config("sensory_anchors")
    if not data:
        # Fallback minimal anchors
        return {
            "anchors": {
                "auditory": ["keyboard clatter nearby"],
                "visual": ["cursor blinking rhythm"],
                "somatic": ["chair lumbar pressure shift"],
                "olfactory": ["stale coffee from mug"],
                "temporal": ["clock check compulsion"],
            },
            "weights": {
                "early_session": {"visual": 0.4, "auditory": 0.35, "somatic": 0.15, "olfactory": 0.05, "temporal": 0.05},
                "mid_session": {"visual": 0.25, "auditory": 0.25, "somatic": 0.30, "olfactory": 0.10, "temporal": 0.10},
                "late_session": {"visual": 0.15, "auditory": 0.15, "somatic": 0.35, "olfactory": 0.10, "temporal": 0.25},
            }
        }
    return data


@lru_cache(maxsize=1)
def get_placeholder_text() -> Dict[str, List[str]]:
    """Load placeholder text components from config.

    Returns:
        Dictionary with intros, bodies, conclusions, etc.
    """
    data = load_config("placeholder_text")
    if not data:
        # Fallback minimal text
        return {
            "intros": ["This empirical investigation demonstrates"],
            "bodies": ["that the underlying assumptions remain foundational."],
            "conclusions": ["Consequently, subsequent research must address these gaps."],
        }
    return data


@lru_cache(maxsize=1)
def get_discourse_markers() -> Dict[str, Any]:
    """Load discourse markers from config.

    Returns:
        Dictionary with repair markers, phase descriptions, etc.
    """
    data = load_config("discourse_markers")
    if not data:
        # Fallback minimal markers
        return {
            "repair_markers": {
                "syntactic_collapse": ["thus, ", "so, ", "consequently, "],
                "lexical_starvation_fallbacks": ["concept", "framework", "element"],
            },
            "phase_descriptions": {
                "Peak": "cognitive resources at maximum capacity",
                "Fatigue": "degraded executive function",
            }
        }
    return data


REQUIRED_CONFIGS = [
    "leakage_patterns",
    "academic_markers",
    "meta_commentary_patterns",
]

OPTIONAL_CONFIGS = [
    "sensory_anchors",
    "discourse_markers",
    "placeholder_text",
    "lexical_substitutions",
    "simulation_config",
]


def validate_configs(config_dir: Optional[Path] = None) -> List[str]:
    """Check that all required config files exist.

    Returns list of missing required config names (empty if all present).
    """
    base_dir = config_dir or CONFIG_DIR
    missing = []
    for name in REQUIRED_CONFIGS:
        path = base_dir / f"{name}.json"
        if not path.exists():
            missing.append(name)
    return missing
