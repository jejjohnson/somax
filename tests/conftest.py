"""Shared pytest configuration for the somax test suite.

Automatic marker assignment
---------------------------
To keep the PR CI job fast, the slow tests are split out via two markers
(declared in ``pyproject.toml``):

- ``integration`` — full-pipeline tests that drive the ``somax-sim``
  runner end to end. Every test under a ``tests/**/test_cli_*.py`` module
  is tagged ``integration`` (and ``slow``).
- ``slow`` — heavy correctness tests dominated by JIT-compiled
  differentiation or long integrations. Every test whose name begins with
  ``test_grad`` is tagged ``slow``, as is every ``integration`` test.

Markers are applied here (rather than by decorating each test) so the
split stays correct as tests are added, without touching every module.

The PR CI job runs ``pytest -m 'not slow'``; the full suite (including
slow + integration) runs in the manual ``full-tests`` workflow.
"""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    """Auto-tag collected tests with ``integration`` / ``slow`` markers."""
    import pytest

    integration = pytest.mark.integration
    slow = pytest.mark.slow

    for item in items:
        filename = item.path.name if hasattr(item, "path") else ""
        is_cli = filename.startswith("test_cli_")
        is_grad = (
            item.originalname.startswith("test_grad")
            if (getattr(item, "originalname", None))
            else item.name.startswith("test_grad")
        )

        if is_cli:
            item.add_marker(integration)
            item.add_marker(slow)
        elif is_grad:
            item.add_marker(slow)
