"""Tests for cognitive calibration module."""
from __future__ import annotations

import math
import pytest
from scholawrite.cognitive_calibration import (
    CognitiveCalibrationReference,
    CALIBRATION_REFERENCES,
    calibrate_config,
    validate_against_references,
)
from scholawrite.config import SimulationConfig


class TestCalibrationReferences:
    """Verify calibration reference data is complete and well-formed."""

    REQUIRED_PARAMETERS = {
        "keystroke_latency_baseline_ms",
        "keystroke_latency_cognitive_load_ms",
        "fatigue_onset_minutes",
        "glucose_depletion_rate",
        "fatigue_divisor",
        "pause_burst_ratio",
        "revision_frequency_per_100_words",
        "syntactic_collapse_glucose_factor",
        "syntactic_collapse_base",
        "lexical_fatigue_penalty",
        "glucose_lexical_starvation",
        "failure_repair_cost_multiplier",
    }

    def test_all_key_parameters_have_sources(self):
        """Every required parameter must have a calibration reference."""
        covered = {ref.parameter for ref in CALIBRATION_REFERENCES}
        missing = self.REQUIRED_PARAMETERS - covered
        assert not missing, f"Missing calibration references for: {missing}"

    def test_references_have_valid_ranges(self):
        """Empirical ranges must be valid (min < max, positive)."""
        for ref in CALIBRATION_REFERENCES:
            lo, hi = ref.empirical_range
            assert lo < hi, f"{ref.parameter}: range min ({lo}) >= max ({hi})"
            assert lo >= 0, f"{ref.parameter}: range min ({lo}) is negative"

    def test_empirical_value_within_range(self):
        """The central empirical value must fall within the range."""
        for ref in CALIBRATION_REFERENCES:
            lo, hi = ref.empirical_range
            assert lo <= ref.empirical_value <= hi, (
                f"{ref.parameter}: empirical_value {ref.empirical_value} "
                f"outside range ({lo}, {hi})"
            )

    def test_all_references_have_citations(self):
        """Every reference must have a non-empty source citation."""
        for ref in CALIBRATION_REFERENCES:
            assert ref.source.strip(), f"{ref.parameter}: empty source citation"
            assert len(ref.source) > 20, f"{ref.parameter}: citation too short"

    def test_all_references_have_notes(self):
        """Every reference must have explanatory notes."""
        for ref in CALIBRATION_REFERENCES:
            assert ref.notes.strip(), f"{ref.parameter}: empty notes"

    def test_reference_is_frozen(self):
        """CognitiveCalibrationReference should be immutable."""
        ref = CALIBRATION_REFERENCES[0]
        with pytest.raises(AttributeError):
            ref.parameter = "modified"  # type: ignore[misc]


class TestCalibrateConfig:
    """Test that calibrate_config produces a valid SimulationConfig."""

    def test_returns_simulation_config(self):
        config = calibrate_config()
        assert isinstance(config, SimulationConfig)

    def test_glucose_starts_at_one(self):
        config = calibrate_config()
        assert config.initial_glucose == 1.0

    def test_glucose_floor_positive(self):
        config = calibrate_config()
        assert 0.0 < config.glucose_floor < 0.5

    def test_depletion_rate_in_empirical_range(self):
        config = calibrate_config()
        assert 0.9988 <= config.glucose_depletion_rate <= 0.9995

    def test_fatigue_divisor_in_empirical_range(self):
        config = calibrate_config()
        assert 8000.0 <= config.fatigue_divisor <= 16000.0

    def test_syntactic_factor_in_range(self):
        config = calibrate_config()
        assert 2.5 <= config.syntactic_collapse_glucose_factor <= 4.5

    def test_different_session_lengths(self):
        """Shorter sessions should produce faster depletion rates."""
        short = calibrate_config(session_duration_minutes=45)
        long = calibrate_config(session_duration_minutes=120)
        # Shorter session -> lower rate (faster depletion per token)
        assert short.glucose_depletion_rate < long.glucose_depletion_rate

    def test_calibrated_glucose_reaches_starvation(self):
        """Glucose should cross 0.65 around the expected fatigue onset."""
        config = calibrate_config(session_duration_minutes=90)
        glucose = 1.0
        tokens_at_onset = int(90 * 15 * 0.58)  # ~783 tokens
        for _ in range(tokens_at_onset):
            glucose *= config.glucose_depletion_rate
        assert 0.55 <= glucose <= 0.75, (
            f"Glucose at fatigue onset = {glucose:.4f}, expected near 0.65"
        )


class TestValidateAgainstReferences:
    """Test validation of configs against empirical references."""

    def test_default_config_passes_within_2x(self):
        """Default SimulationConfig should pass validation within 2x of empirical ranges."""
        results = validate_against_references(SimulationConfig())
        for r in results:
            if r["status"] == "info":
                continue  # Non-mapped parameters
            assert r["status"] in ("pass", "warn"), (
                f"{r['parameter']}: status={r['status']}, "
                f"config_value={r['config_value']}, "
                f"empirical_range={r['empirical_range']}"
            )

    def test_default_config_has_no_failures(self):
        """No parameter in the default config should fail validation."""
        results = validate_against_references(SimulationConfig())
        failures = [r for r in results if r["status"] == "fail"]
        assert not failures, (
            f"Validation failures: "
            + ", ".join(f"{r['parameter']}={r['config_value']}" for r in failures)
        )

    def test_returns_list_of_dicts(self):
        results = validate_against_references(SimulationConfig())
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, dict)
            assert "parameter" in r
            assert "status" in r

    def test_mapped_params_have_deviation(self):
        """Mapped parameters should have a numeric deviation_pct."""
        results = validate_against_references(SimulationConfig())
        mapped = [r for r in results if r["config_field"] is not None]
        assert len(mapped) >= 7, "Expected at least 7 mapped parameters"
        for r in mapped:
            assert r["deviation_pct"] is not None
            assert isinstance(r["deviation_pct"], (int, float))

    def test_extreme_config_fails(self):
        """A wildly off config should produce failures."""
        bad = SimulationConfig(
            glucose_depletion_rate=0.5,
            fatigue_divisor=1.0,
            lexical_fatigue_penalty=100.0,
        )
        results = validate_against_references(bad)
        failures = [r for r in results if r["status"] == "fail"]
        assert len(failures) >= 2, "Expected failures for extreme parameters"

    def test_validation_uses_loaded_config_when_none(self):
        """When no config is passed, validate_against_references uses get_sim_config()."""
        results = validate_against_references()
        assert isinstance(results, list)
        assert len(results) > 0
