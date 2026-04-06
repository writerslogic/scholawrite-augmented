"""Clarkson II loader stub (101 users, 2.5-year longitudinal, gated access).

Source: https://citer.clarkson.edu/clarkson-university-keystroke-dataset-ii/
License: Research-only (CITeR account required)
Citation: Murphy et al. (2017). IJCB.
"""

from __future__ import annotations

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


class Clarkson2Loader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Clarkson II",
        short_name="clarkson2",
        url="https://citer.clarkson.edu/clarkson-university-keystroke-dataset-ii/",
        license="Research-only (CITeR account)",
        access=AccessType.REQUEST,
        citation="Murphy et al. (2017). Shared dataset on natural human-computer interaction. IJCB.",
        description="101 users, 2.5-year longitudinal free-text. 3-col format: timestamp (100ns), event_type, keycode.",
        task_type=TaskType.MIXED,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        raise DatasetAccessRequired(
            "Clarkson II requires a CITeR account.\n"
            "1. Register at https://citer.clarkson.edu/\n"
            "2. Request access to 'Clarkson University Keystroke Dataset II'\n"
            "3. Place the user text files in: " + str(data_dir / "clarkson2")
        )

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "clarkson2"
        txt_files = list(dest.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"Clarkson II data not found in {dest}")

        records: List[CheckpointRecord] = []
        for txt_path in sorted(txt_files):
            cp = self._load_one(txt_path)
            if cp is not None:
                records.append(cp)
        return records

    def _load_one(self, path: Path) -> CheckpointRecord | None:
        press_times: list[float] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    ts = float(parts[0]) * 0.0001
                    event = int(parts[1])
                except (ValueError, IndexError):
                    continue
                if event == 1:
                    press_times.append(ts)

        if len(press_times) < 10:
            return None
        press_times.sort()
        iki_ms = iki_from_timestamps(press_times)
        return to_checkpoint(iki_ms, session_id=f"clarkson2_{path.stem}")


register(Clarkson2Loader())
