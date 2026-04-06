"""External keystroke dataset loaders for cross-dataset validation.

Provides a unified interface for loading keystroke timing data from 17 external
datasets into CheckpointRecord objects compatible with the validation pipeline.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Protocol, runtime_checkable

from scholawrite.validation import CheckpointRecord


# ---------------------------------------------------------------------------
# Metadata types
# ---------------------------------------------------------------------------


class AccessType(Enum):
    OPEN = "open"
    FREE_ACCOUNT = "free_account"
    REQUEST = "request"


class TaskType(Enum):
    COMPOSITION = "composition"
    TRANSCRIPTION = "transcription"
    PASSWORD = "password"
    MIXED = "mixed"
    REFERENCE = "reference"


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    short_name: str
    url: str
    license: str
    access: AccessType
    citation: str
    description: str
    task_type: TaskType = TaskType.MIXED
    signals: List[str] = field(default_factory=list)


class DatasetAccessRequired(Exception):
    """Raised when a dataset requires manual action to download."""


# ---------------------------------------------------------------------------
# Loader protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetLoader(Protocol):
    METADATA: ClassVar[DatasetMeta]

    def download(self, data_dir: Path) -> Path: ...
    def load(self, data_dir: Path) -> List[CheckpointRecord]: ...


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

PAUSE_THRESHOLD_MS = 2000.0
BURST_THRESHOLD_MS = 200.0
_EPSILON = 1e-12


def iki_from_timestamps(timestamps_ms: List[float]) -> List[float]:
    """Derive IKI sequence from sorted press timestamps (ms)."""
    return [timestamps_ms[i] - timestamps_ms[i - 1] for i in range(1, len(timestamps_ms))]


def entropy_bits(values: List[float], n_bins: int = 10) -> float:
    """Shannon entropy of a value distribution in bits.

    Uses log-scale bins since IKI distributions are approximately log-normal.
    """
    if len(values) < 2:
        return 0.0
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return 0.0
    log_vals = [math.log(v) for v in positive]
    lo, hi = min(log_vals), max(log_vals)
    if hi - lo < _EPSILON:
        return 0.0
    bin_width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for lv in log_vals:
        idx = min(int((lv - lo) / bin_width), n_bins - 1)
        counts[idx] += 1
    total = len(positive)
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def lag1_autocorr(values: List[float]) -> Optional[float]:
    """Lag-1 autocorrelation of a sequence."""
    if len(values) < 3:
        return None
    mean = statistics.mean(values)
    var = sum((v - mean) ** 2 for v in values)
    if var < _EPSILON:
        return None
    cov = sum((values[i] - mean) * (values[i + 1] - mean) for i in range(len(values) - 1))
    return cov / var


def to_checkpoint(
    iki_ms: List[float],
    session_id: str = "external",
    seq: int = 0,
    chars_added: int = 0,
    chars_deleted: int = 0,
    dwell_ms: Optional[List[float]] = None,
) -> Optional[CheckpointRecord]:
    """Aggregate an IKI sequence into a CheckpointRecord."""
    if len(iki_ms) < 5:
        return None

    # Filter extreme outliers (> 30 seconds likely not typing)
    filtered = [v for v in iki_ms if 0 < v < 30_000]
    if len(filtered) < 5:
        return None

    mean_iki = statistics.mean(filtered)
    median_iki = statistics.median(filtered)
    std_iki = statistics.stdev(filtered) if len(filtered) > 1 else 0.0
    ent = entropy_bits(filtered)
    ac = lag1_autocorr(filtered)

    pause_count = sum(1 for v in filtered if v > PAUSE_THRESHOLD_MS)
    burst_count = sum(1 for v in filtered if v < BURST_THRESHOLD_MS)

    total_chars = chars_added + chars_deleted
    rev_density = chars_deleted / total_chars if total_chars > 0 else 0.0

    # WPM estimate: assume ~5 chars per word
    total_time_min = sum(filtered) / 60_000.0
    wpm = (len(filtered) / 5.0) / total_time_min if total_time_min > 0 else 0.0

    return CheckpointRecord(
        session_id=session_id,
        seq=seq,
        mean_iki_ms=mean_iki,
        median_iki_ms=median_iki,
        std_iki_ms=std_iki,
        iki_entropy_bits=ent,
        lag1_autocorrelation=ac,
        pause_count=pause_count,
        burst_count=burst_count,
        planning_pause_count=pause_count,
        translating_burst_count=burst_count,
        revising_delete_burst_count=0,
        chars_added=chars_added,
        chars_deleted=chars_deleted,
        revision_density=rev_density,
        wpm=wpm,
        h_prev="",
        h_content="",
        start_time_ns=0,
        end_time_ns=0,
        event_count=len(filtered),
    )


# ---------------------------------------------------------------------------
# Registry (populated by importing individual loader modules)
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, DatasetLoader] = {}


def register(loader: DatasetLoader) -> DatasetLoader:
    """Register a loader instance in the global registry."""
    REGISTRY[loader.METADATA.short_name] = loader
    return loader


def list_datasets() -> List[DatasetMeta]:
    """Return metadata for all registered datasets."""
    return [loader.METADATA for loader in REGISTRY.values()]


def load_dataset(name: str, data_dir: Path) -> List[CheckpointRecord]:
    """Load a single dataset by short name."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return REGISTRY[name].load(data_dir)


def load_by_task_type(
    data_dir: Path,
    task_types: Optional[List[TaskType]] = None,
) -> Dict[str, List[CheckpointRecord]]:
    """Load datasets filtered by task type."""
    if task_types is None:
        task_types = [TaskType.COMPOSITION, TaskType.TRANSCRIPTION, TaskType.MIXED]
    results = {}
    for name, loader in REGISTRY.items():
        if loader.METADATA.task_type not in task_types:
            continue
        try:
            records = loader.load(data_dir)
            if records:
                results[name] = records
        except (FileNotFoundError, DatasetAccessRequired):
            continue
        except Exception:
            continue
    return results


def load_all(data_dir: Path) -> Dict[str, List[CheckpointRecord]]:
    """Load all available datasets, skipping those not downloaded."""
    results = {}
    for name, loader in REGISTRY.items():
        try:
            records = loader.load(data_dir)
            if records:
                results[name] = records
        except (FileNotFoundError, DatasetAccessRequired):
            continue
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Import all loaders to populate REGISTRY
# ---------------------------------------------------------------------------

def _register_all() -> None:
    """Import all loader modules to trigger registration."""
    from scholawrite.datasets import (  # noqa: F401
        aalto_136m,
        aalto_howwetype,
        clarkson2,
        cmu_benchmark,
        early_sci_rev,
        emosurv,
        fastest_typists,
        iiitd_bu,
        ikdd,
        ite_typing,
        keyrecs,
        klicke,
        loggerman,
        mendeley_liveness,
        msu_typing,
        sbu,
        scholawrite_hf,
        stress_detection,
        student_fatigue,
        suny_buffalo,
        synthetic_liveness,
        tappy_parkinsons,
    )


_register_all()
