"""Tests for external dataset loaders.

Uses synthetic fixture data to test each loader's parsing logic
without network access or real dataset files.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from scholawrite.datasets import (
    REGISTRY,
    AccessType,
    DatasetAccessRequired,
    entropy_bits,
    iki_from_timestamps,
    lag1_autocorr,
    list_datasets,
    load_all,
    load_dataset,
    to_checkpoint,
)
from scholawrite.validation import CheckpointRecord


class TestRegistry:
    def test_all_registered(self):
        assert len(REGISTRY) == 22

    def test_list_datasets(self):
        metas = list_datasets()
        assert len(metas) == 22
        names = {m.short_name for m in metas}
        assert "klicke" in names
        assert "cmu_benchmark" in names
        assert "aalto_136m" in names

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            load_dataset("nonexistent", Path("/tmp"))

    def test_metadata_fields(self):
        for loader in REGISTRY.values():
            m = loader.METADATA
            assert m.name
            assert m.short_name
            assert m.url
            assert m.license
            assert isinstance(m.access, AccessType)
            assert m.citation


class TestUtilities:
    def test_iki_from_timestamps(self):
        ts = [100.0, 250.0, 400.0, 700.0]
        iki = iki_from_timestamps(ts)
        assert iki == [150.0, 150.0, 300.0]

    def test_iki_from_timestamps_empty(self):
        assert iki_from_timestamps([]) == []
        assert iki_from_timestamps([100.0]) == []

    def test_entropy_bits_uniform(self):
        values = list(range(1, 101))
        h = entropy_bits(values, n_bins=10)
        assert 2.0 < h < 3.4

    def test_entropy_bits_constant(self):
        assert entropy_bits([5.0] * 100) == 0.0

    def test_entropy_bits_too_few(self):
        assert entropy_bits([]) == 0.0
        assert entropy_bits([1.0]) == 0.0

    def test_lag1_autocorr_constant(self):
        assert lag1_autocorr([1.0, 1.0, 1.0]) is None

    def test_lag1_autocorr_alternating(self):
        vals = [1.0, -1.0] * 50
        ac = lag1_autocorr(vals)
        assert ac is not None
        assert ac < -0.9

    def test_lag1_autocorr_too_short(self):
        assert lag1_autocorr([1.0, 2.0]) is None

    def test_to_checkpoint_basic(self):
        iki = [200.0, 250.0, 180.0, 300.0, 150.0, 220.0, 190.0, 280.0, 210.0, 240.0]
        cp = to_checkpoint(iki, session_id="test")
        assert cp is not None
        assert isinstance(cp, CheckpointRecord)
        assert cp.session_id == "test"
        assert cp.mean_iki_ms > 0
        assert cp.wpm > 0
        assert cp.event_count == 10

    def test_to_checkpoint_too_few(self):
        assert to_checkpoint([100.0, 200.0], session_id="x") is None

    def test_to_checkpoint_filters_outliers(self):
        iki = [200.0] * 10 + [50_000.0]
        cp = to_checkpoint(iki, session_id="x")
        assert cp is not None
        assert cp.event_count == 10

    def test_to_checkpoint_revision_density(self):
        iki = [200.0] * 20
        cp = to_checkpoint(iki, session_id="x", chars_added=80, chars_deleted=20)
        assert cp is not None
        assert abs(cp.revision_density - 0.2) < 0.01


class TestKLiCKeLoader:
    def test_load_from_fixture(self, tmp_path):
        csv_dir = tmp_path / "klicke" / "Files" / "WritingTask" / "WritingTask" / "keystrokelogs" / "csv"
        csv_dir.mkdir(parents=True)

        with open(csv_dir / "10001.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["", "DownEventID", "UpEventID", "DownTime", "UpTime", "ActionTime", "DownEvent", "UpEvent", "Cursorposition", "WordCount", "TextChange", "Activity"])
            w.writeheader()
            t = 1000
            for i in range(50):
                t += 200 + (i % 5) * 30
                w.writerow({"": str(i + 1), "DownEventID": str(i), "UpEventID": str(i), "DownTime": str(t), "UpTime": str(t + 100), "ActionTime": "100", "DownEvent": "a", "UpEvent": "a", "Cursorposition": str(i), "WordCount": str(i // 5), "TextChange": "a", "Activity": "Production"})

        records = load_dataset("klicke", tmp_path)
        assert len(records) == 1
        assert records[0].session_id == "klicke_10001"
        assert records[0].mean_iki_ms > 0


class TestCMUBenchmarkLoader:
    def test_load_from_fixture(self, tmp_path):
        dest = tmp_path / "cmu_benchmark"
        dest.mkdir()
        dd_cols = [f"DD.k{i}.k{i+1}" for i in range(10)]
        header = ["subject", "sessionIndex", "rep"] + dd_cols
        with open(dest / "DSL-StrongPasswordData.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for i in range(5):
                row = {"subject": "s001", "sessionIndex": "1", "rep": str(i)}
                for dd in dd_cols:
                    row[dd] = str(0.1 + i * 0.01)
                w.writerow(row)

        records = load_dataset("cmu_benchmark", tmp_path)
        assert len(records) == 5
        assert records[0].session_id.startswith("cmu_s001")


class TestGatedStubs:
    def test_clarkson2_download_raises(self, tmp_path):
        with pytest.raises(DatasetAccessRequired, match="CITeR"):
            REGISTRY["clarkson2"].download(tmp_path)

    def test_suny_buffalo_download_raises(self, tmp_path):
        with pytest.raises(DatasetAccessRequired, match="shambhu"):
            REGISTRY["suny_buffalo"].download(tmp_path)

    def test_msu_typing_download_raises(self, tmp_path):
        with pytest.raises(DatasetAccessRequired, match="cvlab.cse.msu.edu"):
            REGISTRY["msu_typing"].download(tmp_path)

    def test_emosurv_download_raises(self, tmp_path):
        with pytest.raises(DatasetAccessRequired, match="IEEE"):
            REGISTRY["emosurv"].download(tmp_path)


class TestLoadAll:
    def test_empty_dir(self, tmp_path):
        results = load_all(tmp_path)
        assert results == {}
