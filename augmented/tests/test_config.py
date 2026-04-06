"""Tests for scholawrite.config module."""
from __future__ import annotations

import pytest
from scholawrite.config import (
    SimulationConfig,
    get_sim_config,
    load_config,
    get_leakage_patterns,
    get_academic_markers,
    get_academic_markers_flat,
    get_sensory_anchors,
    get_placeholder_text,
    get_discourse_markers,
    validate_configs,
    CONFIG_DIR,
)


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.initial_glucose == 1.0
        assert cfg.glucose_floor > 0
        assert cfg.glucose_depletion_rate < 1.0

    def test_frozen(self):
        cfg = SimulationConfig()
        with pytest.raises(AttributeError):
            cfg.initial_glucose = 0.5

    def test_thresholds_consistent(self):
        cfg = SimulationConfig()
        assert cfg.locality_human_min < cfg.locality_human_max
        assert cfg.temp_min < cfg.temp_max
        assert cfg.top_p_min < cfg.top_p_max


class TestLoadConfig:
    def test_nonexistent_returns_empty(self):
        result = load_config("nonexistent_config_file_xyz")
        assert result == {}

    def test_loads_known_config(self):
        result = load_config("academic_markers")
        # Should load successfully if config exists
        assert isinstance(result, dict)


class TestGetters:
    def test_leakage_patterns_returns_list(self):
        patterns = get_leakage_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_academic_markers_returns_dict(self):
        markers = get_academic_markers()
        assert isinstance(markers, dict)

    def test_academic_markers_flat(self):
        flat = get_academic_markers_flat()
        assert isinstance(flat, list)
        assert all(isinstance(m, str) for m in flat)

    def test_sensory_anchors(self):
        anchors = get_sensory_anchors()
        assert isinstance(anchors, dict)
        assert "anchors" in anchors or len(anchors) > 0

    def test_placeholder_text(self):
        text = get_placeholder_text()
        assert isinstance(text, dict)

    def test_discourse_markers(self):
        markers = get_discourse_markers()
        assert isinstance(markers, dict)


class TestValidateConfigs:
    def test_returns_list(self):
        result = validate_configs()
        assert isinstance(result, list)

    def test_with_valid_dir(self):
        result = validate_configs(CONFIG_DIR)
        assert isinstance(result, list)
