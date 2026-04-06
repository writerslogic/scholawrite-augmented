"""Aalto 'How We Type' loader (keystroke-only, Typing.zip ~733 kB).

Source: https://zenodo.org/record/4034268
License: CC BY-NC 4.0
Citation: Feit et al. (2016). How We Type. CHI.
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

DOWNLOAD_URL = "https://zenodo.org/api/records/4034268/files/Typing.zip/content"
ZIP_NAME = "Typing.zip"


class AaltoHowWeTypeLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Aalto How We Type",
        short_name="aalto_howwetype",
        url="https://zenodo.org/record/4034268",
        license="CC BY-NC 4.0",
        access=AccessType.OPEN,
        citation="Feit, Weir & Oulasvirta (2016). How We Type. CHI.",
        description="Keystroke logs with IKI directly available. Motion capture + eye tracking available separately.",
        task_type=TaskType.TRANSCRIPTION,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "aalto_howwetype"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / ZIP_NAME
        if zip_path.exists():
            return dest
        import httpx

        resp = httpx.get(DOWNLOAD_URL, follow_redirects=True, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "aalto_howwetype"
        data_files = list(dest.rglob("*.txt")) + list(dest.rglob("*.csv"))
        if not data_files:
            raise FileNotFoundError(f"Aalto HWT data not found in {dest}")

        participants: dict[str, list[float]] = {}
        for fpath in data_files:
            with open(fpath, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    iki_col = None
                    for k in ("iki", "IKI", "IKI_ms", "iki_ms"):
                        if k in row:
                            iki_col = k
                            break
                    if iki_col is None:
                        continue
                    try:
                        iki_val = float(row[iki_col])
                    except (ValueError, TypeError):
                        continue
                    pid = row.get("user_id", row.get("participant", fpath.stem))
                    if 0 < iki_val < 30_000:
                        participants.setdefault(pid, []).append(iki_val)

        records: List[CheckpointRecord] = []
        for pid, iki_ms in participants.items():
            cp = to_checkpoint(iki_ms, session_id=f"aaltohwt_{pid}")
            if cp is not None:
                records.append(cp)
        return records


register(AaltoHowWeTypeLoader())
