"""Stress Detection by Keystroke, App & Mouse Changes loader.

Per-user keystroke TSVs with press/release timestamps and stress conditions.

Source: Kaggle
License: Open
"""

from __future__ import annotations

import csv
import zipfile
from datetime import datetime
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


class StressDetectionLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Stress Detection by Keystroke",
        short_name="stress_detection",
        url="https://www.kaggle.com/datasets/",
        license="Open",
        access=AccessType.OPEN,
        citation="Stress Detection by Keystroke, App & Mouse Changes (2021).",
        description="Per-user keystroke TSVs with press/release timestamps under stress/non-stress conditions.",
        task_type=TaskType.COMPOSITION,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        return data_dir / "stress_detection"

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "stress_detection"
        tsv_files = list(dest.rglob("keystrokes.tsv"))

        if not tsv_files:
            for zp in dest.glob("*.zip"):
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(dest)
            tsv_files = list(dest.rglob("keystrokes.tsv"))

        if not tsv_files:
            raise FileNotFoundError(f"Stress detection data not found in {dest}")

        records: List[CheckpointRecord] = []
        for tsv_path in sorted(tsv_files):
            user_dir = tsv_path.parent.name
            records.extend(self._load_user(tsv_path, user_dir))
        return records

    def _load_user(self, path: Path, user_id: str) -> List[CheckpointRecord]:
        press_times: list[float] = []
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    ts_str = row.get("Press_Time", "")
                    if not ts_str:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts_str)
                        press_times.append(dt.timestamp() * 1000.0)
                    except (ValueError, TypeError):
                        continue
        except OSError:
            return []

        if len(press_times) < 20:
            return []

        press_times.sort()
        iki_ms = [press_times[i] - press_times[i - 1] for i in range(1, len(press_times))]
        cp = to_checkpoint(iki_ms, session_id=f"stress_{user_id}")
        if cp is not None:
            return [cp]
        return []


register(StressDetectionLoader())
