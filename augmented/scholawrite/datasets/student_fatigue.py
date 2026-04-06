"""Student Fatigue Detection via Keystroke Dynamics loader.

Pre-computed keystroke features with fatigue labels. Only has summary
statistics (mean dwell, mean flight, WPM), not raw IKI sequences.
Returns empty from load() -- use load_reference() for calibration data.

Source: Kaggle/Zenodo
License: Open
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

from scholawrite.datasets import (
    AccessType,
    DatasetMeta,
    TaskType,
    register,
)
from scholawrite.validation import CheckpointRecord

CSV_NAME = "dataset_kelelahan_keystroke.csv"


@dataclass(frozen=True)
class FatigueReference:
    participant: str
    session: str
    mean_dwell_ms: float
    std_dwell_ms: float
    mean_flight_ms: float
    std_flight_ms: float
    wpm: float
    error_rate: float
    fatigue_label: int
    sleep_quality: int
    hours_studied: float


class StudentFatigueLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Student Fatigue Keystroke Dynamics",
        short_name="student_fatigue",
        url="https://www.kaggle.com/datasets/",
        license="Open",
        access=AccessType.OPEN,
        citation="Student Fatigue Detection via Keystroke Dynamics (2026).",
        description="Pre-computed dwell, flight, WPM with fatigue labels. Reference-only (no raw IKI).",
        task_type=TaskType.REFERENCE,
        signals=[],
    )

    def download(self, data_dir: Path) -> Path:
        return data_dir / "student_fatigue"

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        return []

    def load_reference(self, data_dir: Path) -> List[FatigueReference]:
        dest = data_dir / "student_fatigue"
        csv_path = dest / CSV_NAME
        if not csv_path.exists():
            for zp in dest.glob("*.zip"):
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(dest)
            csv_path = dest / CSV_NAME
        if not csv_path.exists():
            raise FileNotFoundError(f"Student fatigue data not found in {dest}")

        records: List[FatigueReference] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    records.append(FatigueReference(
                        participant=row.get("id_partisipan", ""),
                        session=row.get("id_sesi", ""),
                        mean_dwell_ms=float(row.get("rata_dwell", 0)),
                        std_dwell_ms=float(row.get("std_dwell", 0)),
                        mean_flight_ms=float(row.get("rata_flight", 0)),
                        std_flight_ms=float(row.get("std_flight", 0)),
                        wpm=float(row.get("kecepatan_ketik_wpm", 0)),
                        error_rate=float(row.get("tingkat_error", 0)),
                        fatigue_label=int(row.get("label", 0)),
                        sleep_quality=int(row.get("kualitas_tidur", 0)),
                        hours_studied=float(row.get("jam_belajar_sebelumnya", 0)),
                    ))
                except (ValueError, TypeError):
                    continue
        return records


register(StudentFatigueLoader())
