"""MSU Typing Behavior loader stub (56+30 subjects, multimodal, gated access).

Source: http://cvlab.cse.msu.edu/typing-behavior-dataset.html
License: Non-commercial research use
Citation: Roth, Liu & Metaxas (2014). IEEE T-IP.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar, List

from scholawrite.datasets import (
    AccessType,
    DatasetAccessRequired,
    DatasetMeta,
    TaskType,
    iki_from_timestamps,
    register,
    to_checkpoint,
)
from scholawrite.validation import CheckpointRecord


class MSUTypingLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="MSU Typing Behavior",
        short_name="msu_typing",
        url="http://cvlab.cse.msu.edu/typing-behavior-dataset.html",
        license="Non-commercial research use",
        access=AccessType.REQUEST,
        citation="Roth, Liu & Metaxas (2014). Unconstrained biometric identification. IEEE T-IP.",
        description="Phase 1: 56 subjects (same-day). Phase 2: 30 subjects (multi-day). Includes video + audio.",
        task_type=TaskType.MIXED,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        raise DatasetAccessRequired(
            "MSU Typing dataset requires download from the project page.\n"
            "1. Visit http://cvlab.cse.msu.edu/typing-behavior-dataset.html\n"
            "2. Download the keystroke data (non-commercial use only)\n"
            "3. Place the data files in: " + str(data_dir / "msu_typing")
        )

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "msu_typing"
        data_files = list(dest.rglob("*.csv")) + list(dest.rglob("*.txt"))
        if not data_files:
            raise FileNotFoundError(f"MSU Typing data not found in {dest}")

        records: List[CheckpointRecord] = []
        for fpath in sorted(data_files):
            records.extend(self._load_file(fpath))
        return records

    def _load_file(self, path: Path) -> List[CheckpointRecord]:
        records: List[CheckpointRecord] = []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                participants: dict[str, list[float]] = {}
                for row in reader:
                    pid = ""
                    for k in ("user", "participant", "subject", "User"):
                        if k in row:
                            pid = row[k]
                            break
                    ts = None
                    for k in ("press_time", "timestamp", "down_time", "time"):
                        if k in row:
                            try:
                                ts = float(row[k])
                            except (ValueError, TypeError):
                                continue
                            break
                    if ts is not None:
                        participants.setdefault(pid or path.stem, []).append(ts)

                for pid, timestamps in participants.items():
                    if len(timestamps) < 10:
                        continue
                    timestamps.sort()
                    iki_ms = iki_from_timestamps(timestamps)
                    cp = to_checkpoint(iki_ms, session_id=f"msu_{pid}")
                    if cp is not None:
                        records.append(cp)
        except (OSError, UnicodeDecodeError):
            pass
        return records


register(MSUTypingLoader())
