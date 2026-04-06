"""
ScholaWrite-Augmented: Embodied Causal simulation for scholarly writing.
"""
from __future__ import annotations

from .schema import (
    InjectionLevel, TrajectoryState, AmbiguityFlag, Label,
    AugmentedDocument, AugmentedRevision, InjectionSpan, CausalEvent
)
from .embodied import EmbodiedScholar, CognitiveState
from .augment import build_augmented
from .io import read_augmented_jsonl, write_augmented_jsonl, load_seed_split
from .banner import print_banner, BANNER_LARGE, BANNER_SMALL, VERSION
from .cli import (
    Spinner, ProgressBar, Menu, MenuItem,
    style, success, error, warning, info, dim, bold, header,
    prompt_path, prompt_choice, prompt_confirm, prompt_number,
    HelpFormatter,
)
from .visualization import (
    generate_html_visualization,
    generate_terminal_visualization,
    generate_tex_report,
    get_citation_info,
    CITATION_BIBTEX,
    CITATION_APA,
    CITATION_MLA,
)
from .config import (
    load_config,
    get_leakage_patterns,
    get_academic_markers,
    get_sensory_anchors,
    get_placeholder_text,
    get_discourse_markers,
    CONFIG_DIR,
)
from .models import ModelRegistry, discover_models, ModelInfo
from .agentic import load_meta_commentary_config, ContentGenerationError


def detect(text: str, *, reference_text: str = "") -> dict:
    """Simple detection API: score a text for AI-generated content.

    Returns dict with per-signal scores and a composite score.
    Higher composite = more likely human-written.

    Args:
        text: Text to analyze.
        reference_text: Optional preceding/reference text for NCD comparison.
    """
    from .baselines import (
        perplexity_proxy_score,
        ngram_burstiness_score,
        _compression_discontinuity,
    )

    perplexity = perplexity_proxy_score(text)
    burstiness = ngram_burstiness_score(text)
    ncd = _compression_discontinuity(reference_text, text) if reference_text else None

    # Composite: perplexity (higher entropy = more human-like),
    # burstiness (higher variance = more human-like),
    # NCD (higher distance from reference = more anomalous = less human)
    signals = [perplexity, burstiness]
    if ncd is not None:
        signals.append(1.0 - ncd)  # invert: low NCD = similar = human

    composite = sum(signals) / len(signals)

    result = {
        "composite": round(composite, 4),
        "perplexity_proxy": round(perplexity, 4),
        "ngram_burstiness": round(burstiness, 4),
    }
    if ncd is not None:
        result["ncd"] = round(ncd, 4)
    return result


__version__ = VERSION
__all__ = [
    # Schema
    "InjectionLevel", "TrajectoryState", "AmbiguityFlag", "Label",
    "AugmentedDocument", "AugmentedRevision", "InjectionSpan", "CausalEvent",
    # Core
    "EmbodiedScholar", "CognitiveState", "build_augmented",
    # I/O
    "read_augmented_jsonl", "write_augmented_jsonl", "load_seed_split",
    # Banner
    "print_banner", "BANNER_LARGE", "BANNER_SMALL", "VERSION",
    # CLI utilities
    "Spinner", "ProgressBar", "Menu", "MenuItem",
    "style", "success", "error", "warning", "info", "dim", "bold", "header",
    "prompt_path", "prompt_choice", "prompt_confirm", "prompt_number",
    "HelpFormatter",
    # Visualization
    "generate_html_visualization",
    "generate_terminal_visualization",
    "generate_tex_report",
    "get_citation_info",
    "CITATION_BIBTEX",
    "CITATION_APA",
    "CITATION_MLA",
    # Config
    "load_config",
    "get_leakage_patterns",
    "get_academic_markers",
    "get_sensory_anchors",
    "get_placeholder_text",
    "get_discourse_markers",
    "CONFIG_DIR",
    # Models (dynamic discovery)
    "ModelRegistry", "discover_models", "ModelInfo",
    # Agentic (LLM orchestration)
    "load_meta_commentary_config", "ContentGenerationError",
    # Detection API
    "detect",
]
