"""Fastest Typists in the World loader (Typeracer leaderboard data).

Only has WPM/accuracy stats, not raw IKI sequences. Returns empty
from load() -- use load_reference() for speed ceiling calibration.

Source: Typeracer leaderboard scrape
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


@dataclass(frozen=True)
class SpeedReference:
    username: str
    career_wpm: float
    best_10_wpm: float
    best_race_wpm: float
    races: int
    wins: int


class FastestTypistsLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Fastest Typists in the World",
        short_name="fastest_typists",
        url="https://www.kaggle.com/datasets/",
        license="Open",
        access=AccessType.OPEN,
        citation="Fastest Typists in the World (Typeracer leaderboard, 2021).",
        description="WPM/accuracy stats for expert typists. Reference-only (no raw IKI).",
        task_type=TaskType.REFERENCE,
        signals=[],
    )

    def download(self, data_dir: Path) -> Path:
        return data_dir / "fastest_typists"

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        return []

    def load_reference(self, data_dir: Path) -> List[SpeedReference]:
        dest = data_dir / "fastest_typists"
        csv_files = list(dest.rglob("*.csv"))
        if not csv_files:
            for zp in dest.glob("*.zip"):
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(dest)
            csv_files = list(dest.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Fastest typists data not found in {dest}")

        records: List[SpeedReference] = []
        for csv_path in csv_files:
            if "leader" not in csv_path.name.lower():
                continue
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        records.append(SpeedReference(
                            username=row.get("Racer", ""),
                            career_wpm=float(row.get("Career", 0)),
                            best_10_wpm=float(row.get("Best_10", 0)),
                            best_race_wpm=float(row.get("Best_Race", 0)),
                            races=int(float(row.get("Races", 0))),
                            wins=int(float(row.get("Wins", 0))),
                        ))
                    except (ValueError, TypeError):
                        continue
        return records


register(FastestTypistsLoader())
