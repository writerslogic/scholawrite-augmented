"""Synthetic Keystroke Liveness Detection loader (ARFF, multiple synthesizer variants).

Contains real and synthesized keystroke features using Gaussian, Histogram,
LCBM, NonStationary, Uniform, and Average synthesizers. Directly relevant
to validating our gradient forger and adversarial evaluation framework.

Source: Figshare (xvg5j5z29p)
License: Open
Citation: Ayotte et al. (2020).
"""

from __future__ import annotations

import re
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

HOLD_RE = re.compile(r"H\.\w+")
DD_RE = re.compile(r"DD\.\w+\.\w+")
UD_RE = re.compile(r"UD\.\w+\.\w+")


class SyntheticLivenessLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="Synthetic Keystroke Liveness Detection",
        short_name="synthetic_liveness",
        url="https://figshare.com/",
        license="Open",
        access=AccessType.OPEN,
        citation="Ayotte et al. (2020). Synthetic keystroke liveness detection.",
        description="ARFF files with real + 5 synthesizer variants. Hold time, DD, UD features.",
        task_type=TaskType.MIXED,
        signals=["mean_iki"],
    )

    def download(self, data_dir: Path) -> Path:
        return data_dir / "synthetic_liveness"

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        dest = data_dir / "synthetic_liveness"
        arff_files = list(dest.rglob("*.arff"))

        if not arff_files:
            for zp in dest.rglob("*.zip"):
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(dest)
            arff_files = list(dest.rglob("*.arff"))

        if not arff_files:
            raise FileNotFoundError(f"Synthetic liveness data not found in {dest}")

        records: List[CheckpointRecord] = []
        for arff_path in sorted(arff_files[:200]):
            cp = self._load_arff(arff_path)
            if cp is not None:
                records.append(cp)
        return records

    def _load_arff(self, path: Path) -> CheckpointRecord | None:
        attributes: list[str] = []
        data_started = False
        dd_indices: list[int] = []
        iki_ms: list[float] = []

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    if line.upper().startswith("@ATTRIBUTE"):
                        parts = line.split()
                        if len(parts) >= 2:
                            attributes.append(parts[1])
                    elif line.upper().startswith("@DATA"):
                        data_started = True
                        dd_indices = [i for i, a in enumerate(attributes) if DD_RE.match(a)]
                        if not dd_indices:
                            dd_indices = [i for i, a in enumerate(attributes) if "DD" in a.upper()]
                        continue

                    if data_started:
                        values = line.split(",")
                        for idx in range(len(values)):
                            if idx >= len(attributes):
                                break
                            attr = attributes[idx].upper()
                            if "DD" in attr or "FT" in attr or "FLIGHT" in attr:
                                try:
                                    val = float(values[idx])
                                    if val < 1.0:
                                        val *= 1000.0
                                    if 0 < val < 30_000:
                                        iki_ms.append(val)
                                except (ValueError, TypeError):
                                    continue
        except OSError:
            return None

        return to_checkpoint(iki_ms, session_id=f"synlive_{path.stem}")


register(SyntheticLivenessLoader())
