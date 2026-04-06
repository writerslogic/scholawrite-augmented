"""Smoke tests to verify all modules import successfully."""
from __future__ import annotations

import importlib
import pytest


MODULES = [
    "scholawrite",
    "scholawrite.schema",
    "scholawrite.config",
    "scholawrite.text",
    "scholawrite.ids",
    "scholawrite.time",
    "scholawrite.labels",
    "scholawrite.metrics",
    "scholawrite.embodied",
    "scholawrite.injection",
    "scholawrite.trajectories",
    "scholawrite.anomalies",
    "scholawrite.baselines",
    "scholawrite.harm",
    "scholawrite.generators",
    "scholawrite.prompts",
    "scholawrite.io",
    "scholawrite.annotations",
    "scholawrite.openrouter",
    "scholawrite.models",
    "scholawrite.causal_core",
    "scholawrite.agentic",
    "scholawrite.augment",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Every module in the package should import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_package_version():
    import scholawrite
    assert hasattr(scholawrite, "__version__")
    assert isinstance(scholawrite.__version__, str)


def test_config_dir_exists():
    from scholawrite.config import CONFIG_DIR
    assert CONFIG_DIR.exists(), f"Config directory not found: {CONFIG_DIR}"


def test_config_validation():
    from scholawrite.config import validate_configs
    missing = validate_configs()
    assert missing == [], f"Missing required configs: {missing}"
