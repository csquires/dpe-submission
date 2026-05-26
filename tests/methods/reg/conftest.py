"""pytest configuration for tests/methods/reg/ suite.

marker registration, session-scoped artifacts directory, and auto-marking
of slow tests by name prefix.
"""
from datetime import datetime
from pathlib import Path

import pytest


def pytest_configure(config):
    """register custom markers."""
    config.addinivalue_line(
        "markers",
        "fast: tests that run in CI by default (<10s), no I/O or GPU required"
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that require opt-in (e.g., long hyperparameter sweeps)"
    )
    config.addinivalue_line(
        "markers",
        "watershed: tests that validate end-to-end behavior or regression detection"
    )


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    """session-scoped directory for test artifacts (logs, json diagnostics).

    creates _artifacts/<timestamp>/ under the conftest's directory to persist
    across runs for trend analysis. timestamp is YYYYMMDD_HHMMSS to avoid
    overwriting prior runs.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_base = Path(__file__).parent / "_artifacts"
    artifacts_base.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts_base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def pytest_collection_modifyitems(config, items):
    """auto-mark slow tests by name prefix.

    defensive catch for tests matching test_hp_response*, test_integration*,
    or test_cross_method* that author may have forgotten to mark. explicit
    markers take precedence; hook adds slow if not already present.
    """
    slow_prefixes = ("test_hp_response", "test_integration", "test_cross_method")
    for item in items:
        if any(item.name.startswith(prefix) for prefix in slow_prefixes):
            # check if slow marker already present
            if "slow" not in [m.name for m in item.iter_markers("slow")]:
                item.add_marker(pytest.mark.slow)
