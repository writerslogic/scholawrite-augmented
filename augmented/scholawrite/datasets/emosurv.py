"""EmoSurv loader (124 participants, keystroke + emotion labels).

Source: https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using
License: Open Access (free IEEE DataPort account)
Citation: Dhakal et al. (2023). IEEE DataPort.
"""

from __future__ import annotations

import csv
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


class EmoSurvLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="EmoSurv",
        short_name="emosurv",
        url="https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using",
        license="Open Access (free IEEE DataPort account)",
        access=AccessType.FREE_ACCOUNT,
        citation="EmoSurv: Typing biometric keystroke dynamics dataset with emotion labels. IEEE DataPort.",
        description="124 participants, 5 emotion states. Fixed and free text with pre-computed digraph features.",
        task_type=TaskType.TRANSCRIPTION,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        raise DatasetAccessRequired(
            "EmoSurv requires a free IEEE DataPort account.\n"
            "1. Create account at https://ieee-dataport.org/\n"
            "2. Download from the dataset page\n"
            "3. Extract CSVs to: " + str(data_dir / "emosurv")
        )

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "emosurv"
        csv_files = list(dest.rglob("*.csv"))
        free_csvs = [f for f in csv_files if "free" in f.name.lower()]
        if not free_csvs:
            free_csvs = csv_files
        if not free_csvs:
            raise FileNotFoundError(f"EmoSurv data not found in {dest}")

        participants: dict[str, list[float]] = {}
        for csv_path in free_csvs:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                # EmoSurv uses semicolons and comma-as-decimal (European locale)
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    uid = row.get("userid", row.get("User_Id", row.get("user_id", "")))
                    iki_col = None
                    for k in ("D1D2", "d1d2"):
                        if k in row:
                            iki_col = k
                            break
                    if iki_col is None:
                        continue
                    try:
                        val = float(row[iki_col].replace(",", "."))
                    except (ValueError, TypeError, AttributeError):
                        continue
                    if 0 < val < 30_000:
                        participants.setdefault(uid, []).append(val)

        records: List[CheckpointRecord] = []
        for uid, iki_ms in participants.items():
            cp = to_checkpoint(iki_ms, session_id=f"emosurv_{uid}")
            if cp is not None:
                records.append(cp)
        return records


register(EmoSurvLoader())
