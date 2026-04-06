"""ITE Typing Dataset loader (~55K participants, keystroke-level with ITE labels).

Source: https://zenodo.org/records/12528163
License: CC-BY
Citation: Aalto Speech (2024). Zenodo.
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
    iki_from_timestamps,
    register,
    to_checkpoint,
)
from scholawrite.validation import CheckpointRecord

DOWNLOAD_URL = "https://zenodo.org/records/12528163/files/ite_typing_dataset.zip?download=1"
ZIP_NAME = "ite_typing_dataset.zip"
WINDOW_SIZE = 200


class ITETypingLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="ITE Typing Dataset",
        short_name="ite_typing",
        url="https://zenodo.org/records/12528163",
        license="CC-BY",
        access=AccessType.OPEN,
        citation="Aalto Speech (2024). ITE Typing Dataset. Zenodo. 10.5281/zenodo.12528163.",
        description="46,755 English + 8,661 Finnish participants. Keystroke-level LOG_DATA with ITE labels.",
        task_type=TaskType.TRANSCRIPTION,
        signals=["mean_iki", "entropy", "lag1_autocorr", "revision_density", "wpm"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "ite_typing"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / ZIP_NAME
        if zip_path.exists():
            return dest
        import httpx
        from tqdm import tqdm

        with httpx.stream("GET", DOWNLOAD_URL, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="ITE Typing") as bar:
                for chunk in r.iter_bytes(65536):
                    f.write(chunk)
                    bar.update(len(chunk))
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "ite_typing"
        zip_path = dest / ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"ITE Typing data not found at {zip_path}")

        records: List[CheckpointRecord] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            log_files = [n for n in zf.namelist() if "log" in n.lower() and n.endswith(".csv")]
            if not log_files:
                log_files = [n for n in zf.namelist() if n.endswith(".csv")]

            for fname in log_files:
                with zf.open(fname) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    records.extend(self._parse_log(text, fname))
        return records

    def _parse_log(self, text_stream: io.TextIOWrapper, source: str) -> List[CheckpointRecord]:
        reader = csv.DictReader(text_stream)
        participants: dict[str, list[float]] = {}
        backspaces: dict[str, int] = {}
        total_keys: dict[str, int] = {}

        for row in reader:
            pid = row.get("PARTICIPANT_ID", row.get("participant_id", ""))
            ts_col = None
            for k in ("TIMESTAMP", "timestamp", "time", "TIME"):
                if k in row:
                    ts_col = k
                    break
            if ts_col is None:
                continue
            try:
                ts = float(row[ts_col])
            except (ValueError, TypeError):
                continue
            participants.setdefault(pid, []).append(ts)
            total_keys[pid] = total_keys.get(pid, 0) + 1

            key = row.get("INPUT_KEY", row.get("key", "")).lower()
            if key in ("backspace", "delete"):
                backspaces[pid] = backspaces.get(pid, 0) + 1

        records: List[CheckpointRecord] = []
        for pid, timestamps in participants.items():
            if len(timestamps) < 10:
                continue
            timestamps.sort()
            iki_ms = iki_from_timestamps(timestamps)
            bs = backspaces.get(pid, 0)
            total = total_keys.get(pid, len(timestamps))
            for i in range(0, len(iki_ms) - WINDOW_SIZE, WINDOW_SIZE):
                window = iki_ms[i : i + WINDOW_SIZE]
                cp = to_checkpoint(
                    window,
                    session_id=f"ite_{pid}_{i // WINDOW_SIZE}",
                    chars_added=max(total - bs, 0),
                    chars_deleted=bs,
                )
                if cp is not None:
                    records.append(cp)
                    break
        return records


register(ITETypingLoader())
