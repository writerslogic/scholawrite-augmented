"""Loggerman/Figshare longitudinal loader (2.5M chars, 1 user, 7 months).

Source: https://figshare.com/articles/dataset/13157510
License: CC-BY
Citation: Smeaton & Doherty (2020). arXiv:2010.10144.
"""

from __future__ import annotations

import csv
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

DOWNLOAD_URL = "https://figshare.com/ndownloader/files/24836825"
FILENAME = "loggerman.zip"
WINDOW_SIZE = 200


class LoggermanLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Loggerman (Figshare Longitudinal)",
        short_name="loggerman",
        url="https://figshare.com/articles/dataset/13157510",
        license="CC-BY",
        access=AccessType.FREE_ACCOUNT,
        citation="Smeaton & Doherty (2020). Keystroke dynamics as part of lifelogging. arXiv:2010.10144.",
        description="2.5M characters typed by one user over 7 months. Character + Unix timestamp.",
        task_type=TaskType.MIXED,
        signals=["mean_iki", "entropy", "lag1_autocorr", "revision_density", "wpm"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "loggerman"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / FILENAME
        if zip_path.exists() or any(dest.rglob("*.csv")):
            return dest
        import httpx
        import zipfile
        from tqdm import tqdm

        with httpx.stream("GET", DOWNLOAD_URL, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Loggerman") as bar:
                for chunk in r.iter_bytes(65536):
                    f.write(chunk)
                    bar.update(len(chunk))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "loggerman"
        csv_files = list(dest.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Loggerman data not found in {dest}")

        all_timestamps: list[float] = []
        all_chars: list[str] = []

        for csv_path in csv_files:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        ts = float(row[1])
                    except (ValueError, TypeError):
                        continue
                    all_timestamps.append(ts)
                    all_chars.append(row[0])

        if len(all_timestamps) < WINDOW_SIZE:
            return []

        pairs = sorted(zip(all_timestamps, all_chars))
        timestamps_ms = [t * 1000.0 for t, _ in pairs]
        chars = [c for _, c in pairs]

        records: List[CheckpointRecord] = []
        for i in range(0, len(timestamps_ms) - WINDOW_SIZE, WINDOW_SIZE):
            window_ts = timestamps_ms[i : i + WINDOW_SIZE]
            window_chars = chars[i : i + WINDOW_SIZE]
            iki_ms = iki_from_timestamps(window_ts)

            backspaces = sum(1 for c in window_chars if c in ("\b", "\x7f", "backspace"))
            chars_added = len(window_chars) - backspaces
            cp = to_checkpoint(
                iki_ms,
                session_id=f"loggerman_{i // WINDOW_SIZE}",
                chars_added=max(chars_added, 0),
                chars_deleted=backspaces,
            )
            if cp is not None:
                records.append(cp)
        return records


register(LoggermanLoader())
