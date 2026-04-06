"""SUNY Buffalo loader stub (148 participants, gated access).

Source: https://www.buffalo.edu/cubs/research/datasets.html
License: Research-request required
Citation: Sun et al. (2016). IEEE TIFS.
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


class SUNYBuffaloLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="SUNY Buffalo",
        short_name="suny_buffalo",
        url="https://www.buffalo.edu/cubs/research/datasets.html",
        license="Research-request required",
        access=AccessType.REQUEST,
        citation="Sun et al. (2016). Shared keystroke dataset for continuous authentication. IEEE TIFS.",
        description="148 participants, 4 keyboard types, ~2.14M keystrokes. Lab-collected over 3 sessions.",
        task_type=TaskType.MIXED,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        raise DatasetAccessRequired(
            "SUNY Buffalo dataset requires email request.\n"
            "1. Email shambhu@buffalo.edu requesting the keystroke dataset\n"
            "2. Place the data files in: " + str(data_dir / "suny_buffalo")
        )

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "suny_buffalo"
        data_files = list(dest.rglob("*.csv")) + list(dest.rglob("*.txt"))
        if not data_files:
            raise FileNotFoundError(f"SUNY Buffalo data not found in {dest}")

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
                    cp = to_checkpoint(iki_ms, session_id=f"buffalo_{pid}")
                    if cp is not None:
                        records.append(cp)
        except (OSError, UnicodeDecodeError):
            pass
        return records


register(SUNYBuffaloLoader())
