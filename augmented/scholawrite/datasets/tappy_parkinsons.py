"""Tappy Keystroke Data with Parkinson's Patients loader.

Raw keystroke timing from Parkinson's patients and controls.
Validates that the simulation's typing distributions fall within
healthy human range (not neurological-impairment range).

Source: https://physionet.org/content/tappy/1.0.0/
License: Open
Citation: Adams (2017). Tappy Keystroke Data.
"""

from __future__ import annotations

import csv
import zipfile
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


class TappyParkinsonsLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Tappy Keystroke Data (Parkinson's)",
        short_name="tappy_parkinsons",
        url="https://physionet.org/content/tappy/1.0.0/",
        license="Open",
        access=AccessType.OPEN,
        citation="Adams (2017). Tappy Keystroke Data with Parkinson's Patients.",
        description="Keystroke timing from Parkinson's patients and healthy controls. Raw press/release timestamps.",
        task_type=TaskType.MIXED,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        return data_dir / "tappy_parkinsons"

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "tappy_parkinsons"
        txt_files = list(dest.rglob("*.txt"))

        if not txt_files:
            for zp in sorted(dest.rglob("*.zip")):
                try:
                    with zipfile.ZipFile(zp, "r") as zf:
                        zf.extractall(zp.parent)
                except zipfile.BadZipFile:
                    continue
            txt_files = list(dest.rglob("*.txt"))

        if not txt_files:
            raise FileNotFoundError(f"Tappy data not found in {dest}")

        users: dict[str, list[float]] = {}
        for txt_path in txt_files:
            if txt_path.name.startswith(".") or "user" in txt_path.name.lower():
                continue
            self._parse_file(txt_path, users)

        records: List[CheckpointRecord] = []
        for uid, iki_values in users.items():
            if len(iki_values) < 20:
                continue
            cp = to_checkpoint(iki_values, session_id=f"tappy_{uid}")
            if cp is not None:
                records.append(cp)
        return records

    def _parse_file(self, path: Path, users: dict[str, list[float]]) -> None:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 7:
                        continue
                    uid = parts[0]
                    try:
                        hold = float(parts[4])
                        flight = float(parts[6])
                    except (ValueError, IndexError):
                        continue
                    iki = hold + flight
                    if 0 < iki < 30_000:
                        users.setdefault(uid, []).append(iki)
        except OSError:
            pass


register(TappyParkinsonsLoader())
