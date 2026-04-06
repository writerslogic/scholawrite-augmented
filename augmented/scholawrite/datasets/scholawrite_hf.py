"""ScholaWrite HuggingFace dataset loader (upstream, 8 authors, 5 projects).

Source: https://huggingface.co/datasets/minnesotanlp/scholawrite
License: Open
Citation: arXiv:2502.02904
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List

from scholawrite.datasets import (
    AccessType,
    DatasetMeta,
    TaskType,
    iki_from_timestamps,
    register,
    to_checkpoint,
)
from scholawrite.validation import CheckpointRecord

WINDOW_SIZE = 200


class ScholaWriteHFLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="ScholaWrite (HuggingFace)",
        short_name="scholawrite_hf",
        url="https://huggingface.co/datasets/minnesotanlp/scholawrite",
        license="Open",
        access=AccessType.OPEN,
        citation="ScholaWrite (2025). arXiv:2502.02904.",
        description="~49K revision events from 8 authors writing 5 manuscripts in Overleaf. Windowed at 200 events.",
        task_type=TaskType.COMPOSITION,
        signals=["mean_iki", "entropy", "lag1_autocorr", "revision_density", "wpm"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "scholawrite_hf"
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.rglob("*.parquet")):
            return dest
        from datasets import load_dataset

        ds = load_dataset("minnesotanlp/scholawrite", split="train")
        ds.to_parquet(str(dest / "train.parquet"))
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "scholawrite_hf"
        parquet_files = list(dest.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"ScholaWrite HF data not found in {dest}")

        import pandas as pd

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        records: List[CheckpointRecord] = []
        for author_id, group in df.groupby("author"):
            timestamps = sorted(group["timestamp"].dropna().astype(float).tolist())
            if len(timestamps) < WINDOW_SIZE:
                continue

            iki_ms = iki_from_timestamps(timestamps)
            before_texts = group["before text"].fillna("").tolist()
            after_texts = group["after text"].fillna("").tolist()

            total_added = 0
            total_deleted = 0
            for bt, at in zip(before_texts, after_texts):
                if len(at) >= len(bt):
                    total_added += len(at) - len(bt)
                else:
                    total_deleted += len(bt) - len(at)

            cp = to_checkpoint(
                iki_ms,
                session_id=f"scholawrite_hf_{author_id}",
                chars_added=total_added,
                chars_deleted=total_deleted,
            )
            if cp is not None:
                records.append(cp)
        return records


register(ScholaWriteHFLoader())
