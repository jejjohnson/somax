"""Tests for the loguru-backed run-log sink."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from loguru import logger

from somax._src.cli._progress import (
    RUN_LOG_FORMAT,
    RunLogContext,
    start_run_log,
    stop_run_log,
)


@pytest.fixture(autouse=True)
def _isolate_loguru():
    """Reset loguru sinks before/after each test so they don't interfere."""
    logger.remove()
    yield
    logger.remove()


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


class TestStartStop:
    def test_start_creates_run_log_file(self, tmp_path: Path) -> None:
        ctx = start_run_log(
            tmp_path, label="test/run", alive_interval=10.0, enable_alive_thread=False
        )
        try:
            assert isinstance(ctx, RunLogContext)
            assert (tmp_path / "run.log").exists()
        finally:
            stop_run_log(ctx)

    def test_start_writes_a_started_line(self, tmp_path: Path) -> None:
        ctx = start_run_log(
            tmp_path, label="test/run", alive_interval=10.0, enable_alive_thread=False
        )
        try:
            content = (tmp_path / "run.log").read_text()
            assert "started" in content
            assert "test/run" in content
        finally:
            stop_run_log(ctx)

    def test_stop_appends_final_message(self, tmp_path: Path) -> None:
        ctx = start_run_log(
            tmp_path, label="test/run", alive_interval=10.0, enable_alive_thread=False
        )
        stop_run_log(ctx, final_message="finished cleanly")
        content = (tmp_path / "run.log").read_text()
        assert "finished cleanly" in content

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        ctx = start_run_log(
            tmp_path, label="test/run", alive_interval=10.0, enable_alive_thread=False
        )
        stop_run_log(ctx)
        stop_run_log(ctx)  # second call should not raise

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        ctx = start_run_log(
            nested, label="x", alive_interval=10.0, enable_alive_thread=False
        )
        try:
            assert (nested / "run.log").exists()
        finally:
            stop_run_log(ctx)

    def test_invalid_interval_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"alive_interval must be > 0"):
            start_run_log(tmp_path, label="x", alive_interval=0.0)
        with pytest.raises(ValueError, match=r"alive_interval must be > 0"):
            start_run_log(tmp_path, label="x", alive_interval=-1.0)


# ----------------------------------------------------------------------
# Logging behavior
# ----------------------------------------------------------------------


class TestLogging:
    def test_bound_logger_writes_to_run_log_file(self, tmp_path: Path) -> None:
        ctx = start_run_log(
            tmp_path, label="run", alive_interval=10.0, enable_alive_thread=False
        )
        try:
            ctx.log.debug("custom marker line")
        finally:
            stop_run_log(ctx)
        content = (tmp_path / "run.log").read_text()
        assert "custom marker line" in content

    def test_unbound_debug_does_not_appear_in_run_log_file(
        self, tmp_path: Path
    ) -> None:
        """A regular logger.debug (no run_log tag) must not pollute the file."""
        ctx = start_run_log(
            tmp_path, label="run", alive_interval=10.0, enable_alive_thread=False
        )
        try:
            logger.debug("not tagged — should NOT land in run.log")
            ctx.log.debug("tagged — should land in run.log")
        finally:
            stop_run_log(ctx)
        content = (tmp_path / "run.log").read_text()
        assert "tagged" in content
        assert "not tagged" not in content

    def test_label_appears_on_every_line(self, tmp_path: Path) -> None:
        ctx = start_run_log(
            tmp_path, label="my-label", alive_interval=10.0, enable_alive_thread=False
        )
        try:
            ctx.log.debug("first")
            ctx.log.debug("second")
        finally:
            stop_run_log(ctx)
        content = (tmp_path / "run.log").read_text()
        # "started" + "first" + "second" = 3 lines, all with the label
        lines = [line for line in content.splitlines() if line.strip()]
        assert len(lines) >= 3
        for line in lines:
            assert "my-label" in line

    def test_format_constant_is_exposed(self) -> None:
        # Sanity: the format string includes the placeholders we expect.
        assert "{time" in RUN_LOG_FORMAT
        assert "{level" in RUN_LOG_FORMAT
        assert "{extra[label]" in RUN_LOG_FORMAT
        assert "{message}" in RUN_LOG_FORMAT


# ----------------------------------------------------------------------
# Alive thread
# ----------------------------------------------------------------------


class TestAliveThread:
    def test_alive_thread_emits_periodic_lines(self, tmp_path: Path) -> None:
        ctx = start_run_log(tmp_path, label="run", alive_interval=0.1)
        try:
            time.sleep(0.45)  # ~4 ticks
        finally:
            stop_run_log(ctx, final_message="done")
        content = (tmp_path / "run.log").read_text()
        n_alive = content.count("alive (wallclock=")
        assert n_alive >= 2, content

    def test_disabling_alive_thread_skips_periodic_lines(self, tmp_path: Path) -> None:
        ctx = start_run_log(
            tmp_path, label="run", alive_interval=0.05, enable_alive_thread=False
        )
        try:
            time.sleep(0.2)  # would be ~4 ticks if thread were running
        finally:
            stop_run_log(ctx)
        content = (tmp_path / "run.log").read_text()
        assert "alive (wallclock=" not in content
