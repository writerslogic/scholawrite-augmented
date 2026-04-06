"""KeyRecs loader (99 participants, 20 nationalities, digraph latencies).

Source: https://zenodo.org/records/7886743
License: CC-BY-4.0
Citation: Mahar et al. (2023). Data in Brief.
"""

from __future__ import annotations

import csv
import io
import zipfile
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

DOWNLOAD_URL = "https://zenodo.org/api/records/7886743/files-archive"
ZIP_NAME = "keyrecs.zip"


class KeyRecsLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="KeyRecs",
        short_name="keyrecs",
        url="https://zenodo.org/records/7886743",
        license="CC-BY-4.0",
        access=AccessType.OPEN,
        citation="Mahar et al. (2023). KeyRecs: A keystroke dynamics dataset. Data in Brief.",
        description="99 participants, 20 nationalities. Fixed-text and free-text digraph latencies.",
        task_type=TaskType.TRANSCRIPTION,
        signals=["mean_iki"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "keyrecs"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / ZIP_NAME
        if zip_path.exists():
            return dest
        import httpx
        from tqdm import tqdm

        with httpx.stream("GET", DOWNLOAD_URL, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="KeyRecs") as bar:
                for chunk in r.iter_bytes(65536):
                    f.write(chunk)
                    bar.update(len(chunk))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "keyrecs"
        free_text = None
        for candidate in [dest / "free-text.csv", dest / "free_text.csv"]:
            if candidate.exists():
                free_text = candidate
                break
        if free_text is None:
            for f in dest.rglob("free*text*.csv"):
                free_text = f
                break
        if free_text is None:
            raise FileNotFoundError(f"KeyRecs free-text.csv not found in {dest}")

        participants: dict[str, list[float]] = {}
        with open(free_text, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("participant", "")
                for key in row:
                    if key.startswith("DD."):
                        try:
                            val = float(row[key])
                            if val > 0:
                                participants.setdefault(pid, []).append(val * 1000.0)
                        except (ValueError, TypeError):
                            continue

        records: List[CheckpointRecord] = []
        for pid, iki_ms in participants.items():
            cp = to_checkpoint(iki_ms, session_id=f"keyrecs_{pid}")
            if cp is not None:
                records.append(cp)
        return records


register(KeyRecsLoader())
