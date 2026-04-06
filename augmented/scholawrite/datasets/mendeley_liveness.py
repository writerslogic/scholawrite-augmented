"""Mendeley Keystroke Liveness Detection loader (real + 5 forgery methods).

Human samples loaded by default. Forgery variants available via load_adversarial()
for comparing against our gradient forger, not for human validation.

Source: https://data.mendeley.com/datasets/mzm86rcxxd/2
License: CC BY-NC 3.0
Citation: Ayotte et al. (2020). Data in Brief.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import ClassVar, List

from scholawrite.datasets import (
    AccessType,
    DatasetMeta,
    DatasetAccessRequired,
    TaskType,
    register,
    to_checkpoint,
)
from scholawrite.validation import CheckpointRecord


class MendeleyLivenessLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Mendeley Keystroke Liveness Detection",
        short_name="mendeley_liveness",
        url="https://data.mendeley.com/datasets/mzm86rcxxd/2",
        license="CC BY-NC 3.0",
        access=AccessType.FREE_ACCOUNT,
        citation="Ayotte et al. (2020). Fast free-text authentication via instance-based keystroke dynamics. Data in Brief.",
        description="488 users, human-written free-text. 5 forgery methods available separately via load_adversarial().",
        task_type=TaskType.COMPOSITION,
        signals=["mean_iki"],
    )

    def download(self, data_dir: Path) -> Path:
        raise DatasetAccessRequired(
            "Mendeley Liveness dataset may require a free Mendeley account.\n"
            "Download from: https://data.mendeley.com/datasets/mzm86rcxxd/2\n"
            "Extract to: " + str(data_dir / "mendeley_liveness")
        )

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        return self._load_filtered(data_dir, human_only=True)

    def load_adversarial(self, data_dir: Path) -> List[CheckpointRecord]:
        return self._load_filtered(data_dir, human_only=False)

    def _load_filtered(self, data_dir: Path, human_only: bool) -> List[CheckpointRecord]:
        dest = data_dir / "mendeley_liveness"
        csv_files = list(dest.rglob("*.csv"))
        zip_files = list(dest.rglob("*.zip"))
        if not csv_files and zip_files:
            for zp in zip_files:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(dest)
            csv_files = list(dest.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Mendeley Liveness data not found in {dest}")

        if human_only:
            csv_files = [f for f in csv_files if "HUMAN" in f.name]
        else:
            csv_files = [f for f in csv_files if "Synthesizer" in f.name]

        subjects: dict[str, list[float]] = {}
        for csv_path in csv_files:
            parts = csv_path.stem.split("-")
            uid = parts[1] if len(parts) > 1 else csv_path.stem
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        flight = float(row[2])
                    except (ValueError, TypeError):
                        continue
                    if flight < 0 or flight > 30_000:
                        continue
                    subjects.setdefault(uid, []).append(flight)

        records: List[CheckpointRecord] = []
        for sid, iki_ms in subjects.items():
            cp = to_checkpoint(iki_ms, session_id=f"mendeley_{sid}")
            if cp is not None:
                records.append(cp)
        return records


register(MendeleyLivenessLoader())
