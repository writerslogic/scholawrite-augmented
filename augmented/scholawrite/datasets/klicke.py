"""KLiCKe Corpus loader (~5,000 argumentative essays with keystroke logs).

Source: https://github.com/terryyutian/KLiCKe-Corpus
License: MIT
Citation: Tian, Crossley & Van Waes (2025), Journal of Writing Research
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

CSV_SUBDIR = Path("klicke") / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"


class KLiCKeLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="KLiCKe Corpus",
        short_name="klicke",
        url="https://github.com/terryyutian/KLiCKe-Corpus",
        license="MIT",
        access=AccessType.OPEN,
        citation="Tian, Crossley & Van Waes (2025). Journal of Writing Research.",
        description="~5,000 argumentative essays with keystroke logs, quality scores, demographics.",
        task_type=TaskType.COMPOSITION,
        signals=["mean_iki", "entropy", "lag1_autocorr", "revision_density", "wpm"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "klicke"
        if (dest / "Files").exists():
            return dest
        try:
            import gdown
        except ImportError as e:
            raise ImportError("pip install gdown") from e
        gdown.download_folder(
            "https://drive.google.com/drive/folders/13Q02Mvj7T-T1OpZzkEqfHwqM7oI61lLR",
            output=str(dest),
            quiet=False,
        )
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        csv_dir = data_dir / CSV_SUBDIR
        if not csv_dir.exists():
            raise FileNotFoundError(f"KLiCKe data not found at {csv_dir}")

        records: List[CheckpointRecord] = []
        for csv_path in sorted(csv_dir.glob("*.csv")):
            cp = self._load_one(csv_path)
            if cp is not None:
                records.append(cp)
        return records

    def _load_one(self, csv_path: Path) -> CheckpointRecord | None:
        writer_id = csv_path.stem
        iki_ms: list[float] = []
        chars_added = 0
        chars_deleted = 0

        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            prev_down = None
            for row in reader:
                try:
                    down_time = float(row.get("DownTime", 0))
                except (ValueError, TypeError):
                    continue

                activity = row.get("Activity", "")
                text_change = row.get("TextChange", "")

                if text_change and text_change != "NoChange":
                    if activity == "Remove/Cut":
                        chars_deleted += len(text_change)
                    else:
                        chars_added += len(text_change)

                if prev_down is not None and down_time > prev_down:
                    iki = down_time - prev_down
                    if 0 < iki < 30_000:
                        iki_ms.append(iki)
                prev_down = down_time

        return to_checkpoint(
            iki_ms,
            session_id=f"klicke_{writer_id}",
            chars_added=chars_added,
            chars_deleted=chars_deleted,
        )


register(KLiCKeLoader())
