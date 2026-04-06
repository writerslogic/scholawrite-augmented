"""IIITD-BU Keystroke Dataset loader (LLM plagiarism detection via keystroke dynamics).

Raw KD/KU events with Unix timestamps in JSON format. Free-text and fixed-text
conditions: transcribed (copying LLM output) vs paraphrased.

Source: https://github.com/ijcb-2024/keystroke-llm-plagiarism
License: Open (code); datasets referenced require separate access
Citation: arXiv:2511.12468
"""

from __future__ import annotations

import json
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

REPO_URL = "https://github.com/ijcb-2024/keystroke-llm-plagiarism.git"


class IIITDBULoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="IIITD-BU Keystroke Dataset",
        short_name="iiitd_bu",
        url="https://github.com/ijcb-2024/keystroke-llm-plagiarism",
        license="Open (code); source datasets may require separate access",
        access=AccessType.OPEN,
        citation="arXiv:2511.12468. Detecting LLM-assisted academic dishonesty via keystroke dynamics.",
        description="130 participants. Raw KD/KU events. Transcribed vs paraphrased LLM output.",
        task_type=TaskType.COMPOSITION,
        signals=["mean_iki", "entropy", "lag1_autocorr"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "iiitd_bu"
        if dest.exists() and any(dest.rglob("*.txt")) or any(dest.rglob("*.json")):
            return dest
        import subprocess

        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "iiitd_bu"
        data_files = list(dest.rglob("free_data.txt")) + list(dest.rglob("fixed_data.txt"))
        json_files = list(dest.rglob("*.json"))
        all_files = data_files + json_files
        if not all_files:
            raise FileNotFoundError(f"IIITD-BU data not found in {dest}")

        records: List[CheckpointRecord] = []
        for fpath in data_files:
            records.extend(self._load_kd_file(fpath))
        for fpath in json_files:
            records.extend(self._load_json(fpath))
        return records

    def _load_kd_file(self, path: Path) -> List[CheckpointRecord]:
        records: List[CheckpointRecord] = []
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        for uid, raw in data.items():
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            else:
                parsed = raw
            kb = parsed.get("keyboard_data", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(kb, list):
                continue
            press_times: list[float] = []
            for event in kb:
                if isinstance(event, list) and len(event) >= 3 and event[0] == "KD":
                    try:
                        press_times.append(float(event[2]))
                    except (ValueError, TypeError):
                        continue
            if len(press_times) < 10:
                continue
            press_times.sort()
            iki_ms = iki_from_timestamps(press_times)
            cp = to_checkpoint(iki_ms, session_id=f"iiitdbu_{uid[:12]}")
            if cp is not None:
                records.append(cp)
        return records

    def _load_json(self, path: Path) -> List[CheckpointRecord]:
        if "SBU" in path.name or "Buffalo" in path.name:
            return self._load_kd_file(path)
        return []


register(IIITDBULoader())
