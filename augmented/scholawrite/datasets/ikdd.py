"""IKDD loader (164 volunteers, 533 logfiles, privacy-preserving).

Source: https://github.com/MachineLearningVisionRG/IKDD
License: Open (cite paper)
Citation: Mahar et al. (2024). Information, 15(9), 511.
"""

from __future__ import annotations

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

REPO_URL = "https://github.com/MachineLearningVisionRG/IKDD.git"


class IKDDLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="IKDD",
        short_name="ikdd",
        url="https://github.com/MachineLearningVisionRG/IKDD",
        license="Open (cite paper)",
        access=AccessType.OPEN,
        citation="Mahar et al. (2024). IKDD: A keystroke dynamics dataset. Information, 15(9), 511.",
        description="164 volunteers, 533 logfiles (~3500 records each). Dwell + digram latencies.",
        task_type=TaskType.MIXED,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "ikdd"
        if dest.exists() and any(dest.rglob("*.txt")):
            return dest
        import subprocess

        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "ikdd"
        txt_files = list(dest.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"IKDD data not found in {dest}")

        records: List[CheckpointRecord] = []
        for txt_path in sorted(txt_files):
            cp = self._load_one(txt_path)
            if cp is not None:
                records.append(cp)
        return records

    def _load_one(self, path: Path) -> CheckpointRecord | None:
        iki_ms: list[float] = []
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    key_spec = parts[0].strip()
                    key_parts = key_spec.split("-")
                    if len(key_parts) != 2:
                        continue
                    second_key = key_parts[1].strip()
                    try:
                        val = float(parts[1].strip())
                    except ValueError:
                        continue
                    if second_key == "0":
                        continue
                    if 0 < val < 3000:
                        iki_ms.append(val)
        except (OSError, UnicodeDecodeError):
            return None

        return to_checkpoint(iki_ms, session_id=f"ikdd_{path.stem}")


register(IKDDLoader())
