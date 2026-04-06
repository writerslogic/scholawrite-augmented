"""Aalto 136 Million Keystrokes loader (~168K participants, sentence transcription).

Source: https://userinterfaces.aalto.fi/136Mkeystrokes/
License: Free for research with attribution
Citation: Dhakal et al. (2018). CHI.
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile

csv.field_size_limit(sys.maxsize)
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

DOWNLOAD_URL = "https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"
ZIP_NAME = "Keystrokes.zip"


class Aalto136MLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Aalto 136M Keystrokes",
        short_name="aalto_136m",
        url="https://userinterfaces.aalto.fi/136Mkeystrokes/",
        license="Research use with attribution",
        access=AccessType.OPEN,
        citation="Dhakal, Szita, Franzen, Bembe & Oulasvirta (2018). Observations on typing from 136 million keystrokes. CHI.",
        description="136M keystrokes from ~168K volunteers transcribing sentences.",
        task_type=TaskType.TRANSCRIPTION,
        signals=["mean_iki", "entropy", "lag1_autocorr", "wpm"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "aalto_136m"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / ZIP_NAME
        if zip_path.exists():
            return dest
        import httpx
        from tqdm import tqdm

        with httpx.stream("GET", DOWNLOAD_URL, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Aalto 136M") as bar:
                for chunk in r.iter_bytes(65536):
                    f.write(chunk)
                    bar.update(len(chunk))
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "aalto_136m"
        zip_path = dest / ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"Aalto 136M data not found at {zip_path}. Run download() first.")

        records: List[CheckpointRecord] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            keystroke_files = [n for n in zf.namelist() if n.endswith(".txt") and "keystrokes" in n.lower()]
            if not keystroke_files:
                keystroke_files = [n for n in zf.namelist() if n.endswith(".txt")]

            for fname in keystroke_files:
                with zf.open(fname) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    records.extend(self._parse_file(text, fname))
        return records

    def _parse_file(self, text_stream: io.TextIOWrapper, source: str) -> List[CheckpointRecord]:
        reader = csv.reader(text_stream, delimiter="\t")
        header = None
        press_col = 5  # PRESS_TIME default index
        pid_col = 0    # PARTICIPANT_ID
        participants: dict[str, list[float]] = {}

        for row in reader:
            if header is None:
                header = [h.strip() for h in row]
                for i, h in enumerate(header):
                    if h == "PRESS_TIME":
                        press_col = i
                    elif h == "PARTICIPANT_ID":
                        pid_col = i
                continue
            if len(row) <= press_col:
                continue
            participant_id = row[pid_col]
            try:
                press_time = float(row[press_col])
            except (ValueError, IndexError):
                continue
            participants.setdefault(participant_id, []).append(press_time)

        records: List[CheckpointRecord] = []
        for pid, timestamps in participants.items():
            if len(timestamps) < 10:
                continue
            timestamps.sort()
            iki_ms = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
            cp = to_checkpoint(iki_ms, session_id=f"aalto_{pid}")
            if cp is not None:
                records.append(cp)
        return records


register(Aalto136MLoader())
