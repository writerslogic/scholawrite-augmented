"""SBU Keystroke Corpus loader (13K sessions, 1060 participants).

Source: https://www3.cs.stonybrook.edu/~rbanerjee/project-pages/keystrokes/
License: Research use (cite paper)
Citation: Banerjee, Feng, Kang & Choi (2014). EMNLP.
"""

from __future__ import annotations

import re
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

KEYDOWN_RE = re.compile(r"\[(\d+)\]\s+KeyDown")


class SBULoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="SBU Keystroke Corpus",
        short_name="sbu",
        url="https://www3.cs.stonybrook.edu/~rbanerjee/project-pages/keystrokes/keystrokes.html",
        license="Research use (cite paper)",
        access=AccessType.OPEN,
        citation="Banerjee, Feng, Kang & Choi (2014). Keystroke patterns as prosody in digital writings. EMNLP.",
        description="13,000 typing sessions from 1,060 participants. Truthful/deceptive essays.",
        task_type=TaskType.COMPOSITION,
        signals=["mean_iki", "entropy", "lag1_autocorr", "wpm"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "sbu"
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.rglob("*.tsv")) or any(dest.rglob("*.bz2")):
            return dest
        import httpx

        base = "https://www3.cs.stonybrook.edu/~rbanerjee/project-pages/keystrokes"
        for fname in ["restaurant_reviews.tar.bz2", "gun_control.tar.bz2", "gay_marriage.tar.bz2"]:
            url = f"{base}/{fname}"
            out = dest / fname
            if out.exists():
                continue
            resp = httpx.get(url, follow_redirects=True, timeout=120)
            resp.raise_for_status()
            out.write_bytes(resp.content)
        import subprocess

        for bz2 in dest.glob("*.tar.bz2"):
            subprocess.run(["tar", "xjf", str(bz2), "-C", str(dest)], check=True)
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "sbu"
        tsv_files = list(dest.rglob("*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"SBU data not found in {dest}")

        records: List[CheckpointRecord] = []
        for tsv_path in sorted(tsv_files):
            records.extend(self._load_file(tsv_path))
        return records

    def _load_file(self, path: Path) -> List[CheckpointRecord]:
        records: List[CheckpointRecord] = []
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f):
                    parts = line.rstrip("\n").split("\t")
                    if not parts:
                        continue
                    meta = parts[-1] if len(parts) > 1 else parts[0]
                    timestamps = [int(m.group(1)) for m in KEYDOWN_RE.finditer(meta)]
                    if len(timestamps) < 10:
                        continue
                    timestamps.sort()
                    iki_ms = [float(timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
                    cp = to_checkpoint(iki_ms, session_id=f"sbu_{path.stem}_{line_no}")
                    if cp is not None:
                        records.append(cp)
        except OSError:
            pass
        return records


register(SBULoader())
