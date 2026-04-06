"""EarlySciRev loader (578K revision pairs from arXiv LaTeX sources).

This is NOT a keystroke timing dataset. It provides revision pairs
(original vs revised text) for validating revision density models.

Source: https://arxiv.org/abs/2603.28515
License: Open
Citation: Jourdan et al. (2026). EarlySciRev. arXiv:2603.28515.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

from scholawrite.datasets import (
    AccessType,
    DatasetMeta,
    TaskType,
    register,
    to_checkpoint,
)
from scholawrite.validation import CheckpointRecord


@dataclass(frozen=True)
class RevisionPair:
    original: str
    revised: str
    revision_density: float


class EarlySciRevLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="EarlySciRev",
        short_name="early_sci_rev",
        url="https://arxiv.org/abs/2603.28515",
        license="Open",
        access=AccessType.OPEN,
        citation="Jourdan et al. (2026). EarlySciRev: Early scientific revisions from arXiv. arXiv:2603.28515.",
        description="578K revision pairs from arXiv LaTeX. Feeds revision_density analysis only.",
        task_type=TaskType.REFERENCE,
        signals=["revision_density"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "early_sci_rev"
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.rglob("*.parquet")) or any(dest.rglob("*.jsonl")):
            return dest
        from datasets import load_dataset

        ds = load_dataset("taln-ls2n/EarlySciRev", "EarlySciRev_llm_filtered", split="train")
        ds.to_parquet(str(dest / "train.parquet"))
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        return []

    def load_revision_pairs(self, data_dir: Path) -> List[RevisionPair]:
        dest = data_dir / "early_sci_rev"
        parquet_files = list(dest.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"EarlySciRev data not found in {dest}")

        import pandas as pd

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        orig_col = "candidate_comments"
        rev_col = "final_parag"
        if orig_col not in df.columns or rev_col not in df.columns:
            return []

        pairs: List[RevisionPair] = []
        for _, row in df.head(10000).iterrows():
            orig = str(row[orig_col]) if row[orig_col] is not None else ""
            rev = str(row[rev_col]) if row[rev_col] is not None else ""
            if not orig or not rev:
                continue
            total = max(len(orig) + len(rev), 1)
            density = abs(len(rev) - len(orig)) / total
            pairs.append(RevisionPair(original=orig, revised=rev, revision_density=density))
        return pairs


register(EarlySciRevLoader())
