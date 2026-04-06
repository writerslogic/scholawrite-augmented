"""CMU Keystroke Dynamics Benchmark loader (51 users, password typing).

Source: https://www.cs.cmu.edu/~keystroke/
License: Open
Citation: Killourhy & Maxion (2009). IEEE S&P.
"""

from __future__ import annotations

import csv
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

DOWNLOAD_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
FILENAME = "DSL-StrongPasswordData.csv"


class CMUBenchmarkLoader:
    METADATA: ClassVar[DatasetMeta] = DatasetMeta(
        name="CMU Keystroke Dynamics Benchmark",
        short_name="cmu_benchmark",
        url="https://www.cs.cmu.edu/~keystroke/",
        license="Open",
        access=AccessType.OPEN,
        citation="Killourhy & Maxion (2009). Comparing anomaly-detection algorithms for keystroke dynamics. IEEE S&P.",
        description="51 users typing '.tie5Roanl' 400 times each. Pre-computed H/DD/UD features.",
        task_type=TaskType.PASSWORD,
        signals=["mean_iki"],
    )

    def download(self, data_dir: Path) -> Path:
        dest = data_dir / "cmu_benchmark"
        dest.mkdir(parents=True, exist_ok=True)
        csv_path = dest / FILENAME
        if csv_path.exists():
            return dest
        import httpx

        resp = httpx.get(DOWNLOAD_URL, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        csv_path.write_bytes(resp.content)
        return dest

    def load(self, data_dir: Path) -> List[CheckpointRecord]:
        csv_path = data_dir / "cmu_benchmark" / FILENAME
        if not csv_path.exists():
            raise FileNotFoundError(f"CMU data not found at {csv_path}")

        subjects: dict[str, list[list[float]]] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row.get("subject", "")
                dd_cols = [k for k in row if k.startswith("DD.")]
                dd_values = []
                for col in dd_cols:
                    try:
                        val = float(row[col]) * 1000.0
                        dd_values.append(val)
                    except (ValueError, TypeError):
                        continue
                if dd_values:
                    subjects.setdefault(subj, []).append(dd_values)

        records: List[CheckpointRecord] = []
        for subj, sessions in subjects.items():
            for i, dd_values in enumerate(sessions):
                cp = to_checkpoint(dd_values, session_id=f"cmu_{subj}_{i}")
                if cp is not None:
                    records.append(cp)
        return records


register(CMUBenchmarkLoader())
